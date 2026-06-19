import Button from '@/components/ui/Button'
import { BrainCircuitIcon, FileDiffIcon, ShieldCheckIcon } from 'lucide-react'
import type { ReactNode } from 'react'

export type TraceRound = {
  round_id?: string
  focus?: string[]
  proposal_ids?: string[]
  state?: string
}

export type TraceStage = {
  stage?: string
  state?: string
  artifact_keys?: string[]
  proposal_ids?: string[]
}

export type LLMReviewPanelProps = {
  trace: Record<string, any> | null
  report: string
  proposals: string
  issueAnalysis: string
  missingBranchInference: string
  evidenceMap: string
  repairPlan: string
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
  issueAnalysis,
  missingBranchInference,
  evidenceMap,
  repairPlan,
  running,
  onRun
}: LLMReviewPanelProps) {
  const stopReason = typeof trace?.stop_reason === 'string' ? trace.stop_reason : ''
  const rounds = Array.isArray(trace?.rounds) ? (trace.rounds as TraceRound[]) : []
  const stages = Array.isArray(trace?.stages)
    ? trace.stages.map(normalizeTraceStageEntry).filter(isTraceStage)
    : []
  const subtitle = stopReason
    ? `停止原因：${stopReason}。辅助材料，不会自动修改 KG。`
    : '尚未生成 LLM 审阅 trace。辅助材料，不会自动修改 KG。'

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <PanelHeader
          icon={<BrainCircuitIcon className="size-4 text-sky-500" />}
          title="LLM 审阅材料"
          subtitle={subtitle}
        />
        <Button variant="outline" size="sm" disabled={running} onClick={onRun}>
          <BrainCircuitIcon className="size-4" />
          {running ? '运行中' : '运行 LLM 审阅'}
        </Button>
      </div>

      <div className="border-border/70 bg-muted/20 rounded-lg border p-3 text-sm">
        <span className="text-muted-foreground mr-2">停止原因</span>
        <span className="font-medium">{stopReason || '暂无 stop_reason'}</span>
      </div>

      <div className="border-border/70 bg-muted/20 rounded-lg border p-3 text-sm">
        LLM Agent 只生成分析、proposal、证据位置和修复排序；任何会修改
        KG、规则、prompt、workspace 或 WebUI 的操作都必须经过人工批准，不会自动修改 KG。
      </div>

      <div className="space-y-2">
        <h3 className="text-sm font-medium">多阶段 LLM Agent</h3>
        <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
          {TRACE_STAGE_LABELS.map(({ key, label }) => {
            const stage = stages.find((item) => normalizeTraceStage(item.stage) === key)
            return (
              <article key={key} className="border-border/70 rounded-lg border p-3 text-sm">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{label}</span>
                  {stage?.state ? (
                    <span className="bg-muted rounded-md px-2 py-1 text-xs">{stage.state}</span>
                  ) : null}
                </div>
                <RoundField
                  label="artifact key"
                  values={stage?.artifact_keys}
                  emptyText="暂无 artifact key。"
                />
                <RoundField
                  label="proposal ID"
                  values={stage?.proposal_ids}
                  emptyText="暂无 proposal ID。"
                />
              </article>
            )
          })}
        </div>
      </div>

      <ArtifactBlock title="问题解释" content={issueAnalysis} emptyText="暂无问题解释。" />
      <ArtifactBlock
        title="缺失分支推断"
        content={missingBranchInference}
        emptyText="暂无缺失分支推断。"
      />
      <ArtifactBlock title="证据定位" content={evidenceMap} emptyText="暂无证据定位。" />
      <ArtifactBlock title="修复方案排序" content={repairPlan} emptyText="暂无修复方案排序。" />

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
              <RoundField label="关注项" values={round.focus} emptyText="暂无关注项。" />
              <RoundField label="proposal ID" values={round.proposal_ids} emptyText="暂无 proposal ID。" />
            </article>
          ))
        ) : (
          <EmptyBlock>暂无审阅轮次。</EmptyBlock>
        )}
      </div>

      <ArtifactBlock title="LLM 审阅报告" content={report} emptyText="暂无审阅报告。" />
      <ArtifactBlock title="生成的 proposal" content={proposals} emptyText="暂无候选 proposal。" />
    </section>
  )
}

const TRACE_STAGE_LABELS = [
  { key: 'explain', label: 'Explain' },
  { key: 'infer', label: 'Infer' },
  { key: 'evidence', label: 'Evidence' },
  { key: 'propose', label: 'Propose' },
  { key: 'rank', label: 'Rank' },
  { key: 'judge', label: 'Judge' }
]

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
        subtitle="从 proposal 加载 Patch 候选，仅供人工检查，不会自动应用。"
      />

      <div className="space-y-2">
        <h3 className="text-sm font-medium">proposal ID</h3>
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
          <EmptyBlock>暂无可加载的 proposal ID。</EmptyBlock>
        )}
      </div>

      <ArtifactBlock title="已选择 Patch" content={patchText} emptyText="尚未选择候选 Patch。" />
      <ArtifactBlock title="proposal 来源" content={proposals} emptyText="暂无候选 proposal。" />
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
      <ArtifactBlock title="Judge 报告" content={report} emptyText="暂无 Judge 评判报告。" />
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

function isTraceStage(stage: TraceStage | null): stage is TraceStage {
  return stage !== null
}

function normalizeTraceStageEntry(stage: unknown): TraceStage | null {
  if (!isRecord(stage)) {
    return null
  }

  return {
    stage: typeof stage.stage === 'string' ? stage.stage : undefined,
    state: typeof stage.state === 'string' ? stage.state : undefined,
    artifact_keys: stringArray(stage.artifact_keys),
    proposal_ids: stringArray(stage.proposal_ids)
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function stringArray(value: unknown) {
  if (!Array.isArray(value)) {
    return undefined
  }

  const strings = value.filter((item): item is string => typeof item === 'string')
  return strings.length ? strings : undefined
}

function normalizeTraceStage(stage?: string) {
  return typeof stage === 'string' ? stage.trim().toLowerCase() : ''
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
