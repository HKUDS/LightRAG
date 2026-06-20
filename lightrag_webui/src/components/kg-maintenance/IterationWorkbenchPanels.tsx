import Button from '@/components/ui/Button'
import type { KBIterationSummaryResponse } from '@/api/lightrag'
import type { KGMaintenanceSection } from '@/stores/kgMaintenance'
import {
  ArchiveIcon,
  CheckCircle2Icon,
  ClipboardListIcon,
  FileJsonIcon,
  FileTextIcon,
  GitPullRequestIcon,
  ListChecksIcon,
  PlayCircleIcon,
  ShieldAlertIcon,
  TimerIcon,
  XCircleIcon
} from 'lucide-react'
import type { ReactNode } from 'react'
import { SnapshotTables } from './SnapshotTables'

type IterationOverviewPanelProps = {
  summary: KBIterationSummaryResponse | null
  loading: boolean
  onOpenSection: (section: KGMaintenanceSection) => void
}

type MarkdownArtifactPanelProps = {
  icon?: ReactNode
  title: string
  fileName: string
  content: string
  emptyText: string
}

type JsonArtifactPanelProps = {
  title: string
  fileName: string
  payload: unknown
  summaryRows: Array<[string, string]>
  emptyText: string
}

type IterationReviewAsideProps = {
  phase?: string
  pendingApprovalCount?: number
  highRiskFindingCount?: number
}

const requiredArtifacts: Array<{
  key: string
  label: string
  fileName: string
  section: KGMaintenanceSection
  icon: ReactNode
}> = [
  {
    key: 'kb_context',
    label: '当前 KB 摘要',
    fileName: 'kb_context.md',
    section: 'check',
    icon: <FileTextIcon className="size-4 text-sky-500" />
  },
  {
    key: 'quality_report',
    label: '质量报告',
    fileName: 'quality_report.md',
    section: 'check',
    icon: <ShieldAlertIcon className="size-4 text-amber-500" />
  },
  {
    key: 'kg_snapshot',
    label: '图谱快照',
    fileName: 'snapshots/kg_snapshot.json',
    section: 'check',
    icon: <ArchiveIcon className="size-4 text-indigo-500" />
  },
  {
    key: 'quality_score',
    label: '质量分数',
    fileName: 'snapshots/quality_score.json',
    section: 'check',
    icon: <FileJsonIcon className="size-4 text-emerald-500" />
  },
  {
    key: 'approval_queue',
    label: '待审批 Proposal',
    fileName: 'approval_queue.md',
    section: 'approval',
    icon: <GitPullRequestIcon className="size-4 text-violet-500" />
  },
  {
    key: 'improvement_backlog',
    label: '改进 backlog',
    fileName: 'improvement_backlog.md',
    section: 'execute',
    icon: <ClipboardListIcon className="size-4 text-cyan-600" />
  },
  {
    key: 'accepted_changes',
    label: '已接受变更记忆',
    fileName: 'accepted_changes.md',
    section: 'execute',
    icon: <CheckCircle2Icon className="size-4 text-emerald-600" />
  },
  {
    key: 'rejected_changes',
    label: '已拒绝变更记忆',
    fileName: 'rejected_changes.md',
    section: 'execute',
    icon: <XCircleIcon className="size-4 text-rose-500" />
  },
  {
    key: 'agent_memory_summary',
    label: 'Agent 压缩记忆',
    fileName: 'agent_memory_summary.md',
    section: 'approval',
    icon: <FileTextIcon className="size-4 text-teal-600" />
  },
  {
    key: 'accepted_changes_apply_result',
    label: '真实应用结果',
    fileName: 'accepted_changes_apply_result.md',
    section: 'execute',
    icon: <CheckCircle2Icon className="size-4 text-emerald-600" />
  },
  {
    key: 'accepted_changes_execution',
    label: '执行报告',
    fileName: 'accepted_changes_execution.md',
    section: 'execute',
    icon: <PlayCircleIcon className="size-4 text-indigo-600" />
  },
  {
    key: 'iteration_log',
    label: '当前阶段',
    fileName: 'iteration_log.md',
    section: 'validate',
    icon: <TimerIcon className="size-4 text-slate-500" />
  }
]

export function IterationOverviewPanel({
  summary,
  loading,
  onOpenSection
}: IterationOverviewPanelProps) {
  if (loading && !summary) {
    return (
      <section className="space-y-4">
        <div>
          <h2 className="text-sm font-semibold">知识库迭代 Agent</h2>
          <p className="text-muted-foreground mt-1 text-sm">正在读取迭代产物。</p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className="border-border/70 rounded-lg border p-3">
              <div className="bg-muted h-3 w-20 rounded" />
              <div className="bg-muted mt-3 h-5 w-28 rounded" />
            </div>
          ))}
        </div>
        <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, index) => (
            <div key={index} className="border-border/70 rounded-lg border p-3">
              <div className="bg-muted h-4 w-32 rounded" />
              <div className="bg-muted mt-3 h-3 w-44 rounded" />
            </div>
          ))}
        </div>
      </section>
    )
  }

  if (!summary) {
    return (
      <section className="border-border/70 bg-muted/20 rounded-lg border p-5">
        <h2 className="text-sm font-semibold">知识库迭代 Agent</h2>
        <p className="text-muted-foreground mt-2 text-sm">
          请选择 workspace，并运行 review package 后查看当前 KB 摘要、质量报告、快照和审批产物。
        </p>
      </section>
    )
  }

  const artifactExists = new Map(
    summary.artifacts.map((artifact) => [artifact.key, artifact.exists])
  )

  return (
    <section className="space-y-4">
      <div>
        <h2 className="text-sm font-semibold">知识库迭代 Agent</h2>
        <p className="text-muted-foreground mt-1 text-sm">
          {summary.workspace} / {summary.latestRunId}
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricTile label="当前阶段" value={summary.phase || '未开始'} />
        <MetricTile label="质量分数" value={formatMaybeNumber(summary.quality.overall)} />
        <MetricTile label="待审批 Proposal" value={String(summary.pendingApprovalCount ?? 0)} />
        <MetricTile
          label="节点 / 关系"
          value={`${summary.counts.nodes ?? 0} / ${summary.counts.edges ?? 0}`}
        />
      </div>

      <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
        {requiredArtifacts.map((artifact) => {
          const exists = artifactExists.get(artifact.key) ?? false
          return (
            <button
              key={artifact.key}
              type="button"
              onClick={() => onOpenSection(artifact.section)}
              className="border-border/70 bg-background hover:bg-muted/30 focus-visible:ring-ring rounded-lg border p-3 text-left transition-colors focus-visible:ring-2 focus-visible:outline-none"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex min-w-0 items-start gap-2">
                  {artifact.icon}
                  <div className="min-w-0">
                    <div className="text-sm font-medium">{artifact.label}</div>
                    <div className="text-muted-foreground mt-1 text-xs break-words">
                      {artifact.fileName}
                    </div>
                  </div>
                </div>
                <span
                  className={
                    exists
                      ? 'rounded-md bg-emerald-50 px-2 py-1 text-xs text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-200'
                      : 'bg-muted text-muted-foreground rounded-md px-2 py-1 text-xs'
                  }
                >
                  {exists ? '已生成' : '缺失'}
                </span>
              </div>
            </button>
          )
        })}
      </div>
    </section>
  )
}

export function IterationStagePanel({ iterationLog }: { iterationLog: string }) {
  return (
    <MarkdownArtifactPanel
      icon={<TimerIcon className="size-4 text-slate-500" />}
      title="当前阶段"
      fileName="iteration_log.md"
      content={iterationLog}
      emptyText="暂无当前阶段记录。"
    />
  )
}

export function KBSummaryPanel({ kbContext }: { kbContext: string }) {
  return (
    <MarkdownArtifactPanel
      icon={<FileTextIcon className="size-4 text-sky-500" />}
      title="当前 KB 摘要"
      fileName="kb_context.md"
      content={kbContext}
      emptyText="暂无 KB 摘要。"
    />
  )
}

export function BacklogPanel({ improvementBacklog }: { improvementBacklog: string }) {
  return (
    <MarkdownArtifactPanel
      icon={<ClipboardListIcon className="size-4 text-cyan-600" />}
      title="改进 backlog"
      fileName="improvement_backlog.md"
      content={improvementBacklog}
      emptyText="暂无改进 backlog。"
    />
  )
}

export function DecisionMemoryPanel({
  acceptedChanges,
  rejectedChanges
}: {
  acceptedChanges: string
  rejectedChanges: string
}) {
  return (
    <section className="grid gap-3 lg:grid-cols-2">
      <MarkdownArtifactPanel
        icon={<CheckCircle2Icon className="size-4 text-emerald-600" />}
        title="已接受变更记忆"
        fileName="accepted_changes.md"
        content={acceptedChanges}
        emptyText="暂无已接受变更记忆。"
      />
      <MarkdownArtifactPanel
        icon={<XCircleIcon className="size-4 text-rose-500" />}
        title="已拒绝变更记忆"
        fileName="rejected_changes.md"
        content={rejectedChanges}
        emptyText="暂无已拒绝变更记忆。"
      />
    </section>
  )
}

export function DecisionExecutionPanel({
  improvementBacklog,
  acceptedChanges,
  rejectedChanges,
  acceptedApplyResult,
  acceptedExecution,
  executing,
  onExecuteAcceptedChanges
}: {
  improvementBacklog: string
  acceptedChanges: string
  rejectedChanges: string
  acceptedApplyResult: string
  acceptedExecution: string
  executing: boolean
  onExecuteAcceptedChanges: () => void
}) {
  const acceptedCount = countDecisionSections(acceptedChanges)

  return (
    <section className="space-y-3">
      <div className="border-border/70 bg-muted/20 flex flex-wrap items-center justify-between gap-3 rounded-lg border p-3">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold">决策与执行</h2>
          <p className="text-muted-foreground mt-1 text-xs">
            {acceptedCount > 0 ? `${acceptedCount} 条已接受变更等待 Agent 执行` : '暂无已接受变更'}
          </p>
        </div>
        <Button
          type="button"
          size="sm"
          onClick={onExecuteAcceptedChanges}
          disabled={executing || acceptedCount === 0}
        >
          <PlayCircleIcon className="size-4" />
          {executing ? 'Agent 执行中' : 'Agent 执行已接受变更'}
        </Button>
      </div>

      <div className="grid gap-3 xl:grid-cols-2">
        <MarkdownArtifactPanel
          icon={<ClipboardListIcon className="size-4 text-cyan-600" />}
          title="改进 backlog"
          fileName="improvement_backlog.md"
          content={improvementBacklog}
          emptyText="暂无改进 backlog。"
        />
        <MarkdownArtifactPanel
          icon={<CheckCircle2Icon className="size-4 text-emerald-600" />}
          title="真实应用结果"
          fileName="accepted_changes_apply_result.md"
          content={acceptedApplyResult}
          emptyText="暂无真实应用结果。"
        />
        <MarkdownArtifactPanel
          icon={<PlayCircleIcon className="size-4 text-indigo-600" />}
          title="执行报告"
          fileName="accepted_changes_execution.md"
          content={acceptedExecution}
          emptyText="暂无执行报告。"
        />
        <MarkdownArtifactPanel
          icon={<CheckCircle2Icon className="size-4 text-emerald-600" />}
          title="已接受变更记忆"
          fileName="accepted_changes.md"
          content={acceptedChanges}
          emptyText="暂无已接受变更记忆。"
        />
        <MarkdownArtifactPanel
          icon={<XCircleIcon className="size-4 text-rose-500" />}
          title="已拒绝变更记忆"
          fileName="rejected_changes.md"
          content={rejectedChanges}
          emptyText="暂无已拒绝变更记忆。"
        />
      </div>
    </section>
  )
}

export function SnapshotReviewPanel({
  snapshot,
  qualityScore
}: {
  snapshot: unknown
  qualityScore?: unknown
}) {
  const record = asRecord(snapshot)
  const nodeCount = countCollection(record?.nodes) ?? countCollection(record?.entities)
  const relationCount =
    countCollection(record?.edges) ??
    countCollection(record?.relations) ??
    countCollection(record?.links)

  return (
    <section className="border-border/70 rounded-lg border p-3">
      <ArtifactHeader title="图谱快照" fileName="snapshots/kg_snapshot.json" />
      <dl className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
        {[
          ['节点数', formatMaybeNumber(nodeCount)],
          ['关系数', formatMaybeNumber(relationCount)],
          ['Workspace', formatUnknown(record?.workspace)],
          ['Snapshot ID', formatUnknown(record?.snapshot_id ?? record?.snapshotId ?? record?.id)]
        ].map(([label, value]) => (
          <div key={label} className="bg-muted/30 rounded-md p-2">
            <dt className="text-muted-foreground text-xs">{label}</dt>
            <dd className="mt-1 text-sm font-medium break-words">{value}</dd>
          </div>
        ))}
      </dl>
      <div className="mt-3">
        <SnapshotTables snapshot={record} qualityScore={qualityScore} />
      </div>
    </section>
  )
}

export function QualityScoreJsonPanel({ qualityScore }: { qualityScore: unknown }) {
  const record = asRecord(qualityScore)

  return (
    <JsonArtifactPanel
      title="质量分数"
      fileName="snapshots/quality_score.json"
      payload={qualityScore}
      summaryRows={[
        ['Overall', formatUnknown(record?.overall)],
        ['Findings', formatMaybeNumber(countCollection(record?.findings))],
        ['Critical blockers', formatMaybeNumber(countCollection(record?.critical_blockers))]
      ]}
      emptyText="暂无质量分数。"
    />
  )
}

export function JsonArtifactPanel({
  title,
  fileName,
  payload,
  summaryRows,
  emptyText
}: JsonArtifactPanelProps) {
  const hasPayload = payload !== null && payload !== undefined && payload !== ''

  return (
    <section className="border-border/70 rounded-lg border p-3">
      <ArtifactHeader title={title} fileName={fileName} />
      <dl className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
        {summaryRows.map(([label, value]) => (
          <div key={label} className="bg-muted/30 rounded-md p-2">
            <dt className="text-muted-foreground text-xs">{label}</dt>
            <dd className="mt-1 text-sm font-medium break-words">{value}</dd>
          </div>
        ))}
      </dl>
      <pre className="border-border/70 bg-muted/20 text-muted-foreground mt-3 max-h-96 overflow-auto rounded-md border p-3 text-xs break-words whitespace-pre-wrap">
        {hasPayload ? JSON.stringify(payload, null, 2) : emptyText}
      </pre>
    </section>
  )
}

export function MarkdownArtifactPanel({
  icon,
  title,
  fileName,
  content,
  emptyText
}: MarkdownArtifactPanelProps) {
  const hasContent = content.trim().length > 0

  return (
    <section className="border-border/70 rounded-lg border p-3">
      <ArtifactHeader
        icon={icon ?? <FileTextIcon className="size-4 text-sky-500" />}
        title={title}
        fileName={fileName}
      />
      <pre className="text-muted-foreground mt-3 max-h-96 overflow-auto text-xs break-words whitespace-pre-wrap">
        {hasContent ? content : emptyText}
      </pre>
    </section>
  )
}

export function IterationReviewAside({
  phase,
  pendingApprovalCount = 0,
  highRiskFindingCount = 0
}: IterationReviewAsideProps) {
  return (
    <aside className="border-border/70 rounded-lg border p-3">
      <div className="flex items-center gap-2">
        <ListChecksIcon className="size-4 text-indigo-500" />
        <h2 className="text-sm font-semibold">审阅侧栏</h2>
      </div>
      <dl className="mt-3 space-y-2 text-sm">
        <AsideRow label="当前阶段" value={phase || '未开始'} />
        <AsideRow label="待审批 Proposal" value={String(pendingApprovalCount)} />
        <AsideRow label="高风险发现" value={String(highRiskFindingCount)} />
      </dl>
      <div className="border-border/70 bg-muted/20 text-muted-foreground mt-3 rounded-md border p-3 text-xs leading-5">
        所有事实、规则、prompt、rebuild 相关变更都需要人工审批，不自动修改 KG。
      </div>
    </aside>
  )
}

function ArtifactHeader({
  icon,
  title,
  fileName
}: {
  icon?: ReactNode
  title: string
  fileName: string
}) {
  return (
    <div className="flex min-w-0 items-start gap-2">
      {icon ? icon : null}
      <div className="min-w-0">
        <h2 className="text-sm font-semibold">{title}</h2>
        <p className="text-muted-foreground mt-1 text-xs break-words">{fileName}</p>
      </div>
    </div>
  )
}

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-border/70 rounded-lg border p-3">
      <div className="text-muted-foreground text-xs">{label}</div>
      <div className="mt-1 text-lg font-semibold break-words">{value}</div>
    </div>
  )
}

function AsideRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-medium">{value}</dd>
    </div>
  )
}

function asRecord(value: unknown): Record<string, any> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, any>)
    : null
}

function countCollection(value: unknown): number | undefined {
  if (Array.isArray(value)) return value.length
  if (value && typeof value === 'object') return Object.keys(value).length
  if (typeof value === 'number') return value
  return undefined
}

function countDecisionSections(content: string): number {
  return Array.from(content.matchAll(/^##\s+\S+/gm)).length
}

function formatUnknown(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  return JSON.stringify(value)
}

function formatMaybeNumber(value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value) ? String(value) : formatUnknown(value)
}
