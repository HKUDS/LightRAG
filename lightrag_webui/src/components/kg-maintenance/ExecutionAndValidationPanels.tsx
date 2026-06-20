import Button from '@/components/ui/Button'
import { CheckCircle2Icon, PlayCircleIcon, ShieldCheckIcon } from 'lucide-react'
import type { QualityBeforeSnapshot } from './kgMaintenanceDisplay'

type ExecutionPanelProps = {
  acceptedChanges: string
  applyResult: string
  executing: boolean
  onExecute: () => void
}

type ValidationPanelProps = {
  qualityBefore: QualityBeforeSnapshot | null | undefined
  qualityAfter: Record<string, any> | null | undefined
  applyResult: string
}

export function ExecutionPanel({
  acceptedChanges,
  applyResult,
  executing,
  onExecute
}: ExecutionPanelProps) {
  const acceptedCount = countAcceptedChangeHeadings(acceptedChanges)
  const canExecute = acceptedCount > 0 && !executing

  return (
    <section className="space-y-3">
      <div className="border-border/70 bg-muted/20 flex flex-wrap items-center justify-between gap-3 rounded-lg border p-3">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold">执行已接受变更</h2>
          <p className="text-muted-foreground mt-1 text-xs">
            {acceptedCount > 0
              ? `${acceptedCount} 条已接受变更等待写入`
              : '暂无可执行的已接受变更'}
          </p>
        </div>
        <Button type="button" size="sm" onClick={onExecute} disabled={!canExecute}>
          <PlayCircleIcon className="size-4" />
          {executing ? '执行中' : '执行变更'}
        </Button>
      </div>

      <section className="border-border/70 rounded-lg border p-3">
        <div className="flex min-w-0 items-start gap-2">
          <CheckCircle2Icon className="size-4 text-emerald-600" />
          <div className="min-w-0">
            <h3 className="text-sm font-semibold">应用结果</h3>
            <p className="text-muted-foreground mt-1 text-xs">
              {formatAppliedLine(applyResult) || '尚未执行写入'}
            </p>
          </div>
        </div>
        <pre className="text-muted-foreground mt-3 max-h-80 overflow-auto text-xs break-words whitespace-pre-wrap">
          {applyResult.trim() || '暂无应用结果。'}
        </pre>
      </section>

      <section className="border-border/70 rounded-lg border p-3">
        <h3 className="text-sm font-semibold">已接受变更</h3>
        <pre className="text-muted-foreground mt-3 max-h-80 overflow-auto text-xs break-words whitespace-pre-wrap">
          {acceptedChanges.trim() || '暂无已接受变更。'}
        </pre>
      </section>
    </section>
  )
}

export function ValidationPanel({
  qualityBefore,
  qualityAfter,
  applyResult
}: ValidationPanelProps) {
  const overallBefore = qualityBefore?.overall
  const overallAfter = readQualityValue(qualityAfter, 'overall')
  const missingBefore = qualityBefore?.metrics?.hierarchy_missing_branch_count
  const missingAfter = readQualityValue(qualityAfter, 'hierarchy_missing_branch_count')
  const alreadyMeetsTarget = isAppliedZero(applyResult) && missingAfter === 0

  return (
    <section className="space-y-3">
      <div className="border-border/70 bg-muted/20 rounded-lg border p-3">
        <div className="flex min-w-0 items-start gap-2">
          <ShieldCheckIcon className="size-4 text-emerald-600" />
          <div className="min-w-0">
            <h2 className="text-sm font-semibold">验证结果</h2>
            <p className="text-muted-foreground mt-1 text-xs">
              对比执行前后的质量指标，确认已接受变更是否达成目标。
            </p>
          </div>
        </div>
      </div>

      {alreadyMeetsTarget ? (
        <div className="border-border/70 bg-emerald-50/70 rounded-lg border p-3 text-sm text-emerald-800 dark:bg-emerald-950/30 dark:text-emerald-200">
          没有新增写入，但当前质量已达标
        </div>
      ) : null}

      <dl className="grid gap-2 sm:grid-cols-2">
        <DeltaRow label="总分" before={overallBefore} after={overallAfter} />
        <DeltaRow label="缺失分支" before={missingBefore} after={missingAfter} />
      </dl>

      <section className="border-border/70 rounded-lg border p-3">
        <h3 className="text-sm font-semibold">应用结果</h3>
        <pre className="text-muted-foreground mt-3 max-h-80 overflow-auto text-xs break-words whitespace-pre-wrap">
          {applyResult.trim() || '暂无应用结果。'}
        </pre>
      </section>
    </section>
  )
}

function DeltaRow({
  label,
  before,
  after
}: {
  label: string
  before: unknown
  after: unknown
}) {
  return (
    <div className="border-border/70 rounded-lg border p-3">
      <dt className="text-muted-foreground text-xs">{label}</dt>
      <dd className="mt-1 text-sm font-semibold">
        {formatQualityValue(before)} → {formatQualityValue(after)}
      </dd>
    </div>
  )
}

function countAcceptedChangeHeadings(content: string): number {
  return Array.from(content.matchAll(/^##\s+[A-Za-z0-9][A-Za-z0-9_.-]*\s*$/gm)).length
}

function formatAppliedLine(applyResult: string): string {
  return applyResult
    .split(/\r?\n/)
    .map((line) => line.trim())
    .map((line) => line.match(/^(?:-\s*)?(Applied:\s*\d+)/i)?.[1] ?? '')
    .find(Boolean) ?? ''
}

function isAppliedZero(applyResult: string): boolean {
  return /^\s*(?:-\s*)?Applied:\s*0\b/im.test(applyResult)
}

function readQualityValue(quality: Record<string, any> | null | undefined, key: string): unknown {
  if (!quality) return undefined
  if (key in quality) return quality[key]
  const metrics = quality.metrics
  if (metrics && typeof metrics === 'object' && !Array.isArray(metrics) && key in metrics) {
    return metrics[key]
  }
  return undefined
}

function formatQualityValue(value: unknown): string {
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  if (typeof value === 'string' && value.trim().length > 0) return value
  return '—'
}
