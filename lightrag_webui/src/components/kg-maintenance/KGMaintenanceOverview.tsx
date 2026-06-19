import Progress from '@/components/ui/Progress'
import type { KBIterationSummaryResponse } from '@/api/lightrag'
import { formatRunSubtitle, getEvidenceCoveragePercent } from './kgMaintenanceData'
import type { KGMaintenanceSection } from '@/stores/kgMaintenance'

interface KGMaintenanceOverviewProps {
  summary: KBIterationSummaryResponse | null
  loading: boolean
  onOpenSection: (section: KGMaintenanceSection) => void
}

const metricClass =
  'rounded-lg border border-border/70 bg-background p-3 transition-colors hover:bg-accent/40'

export default function KGMaintenanceOverview({
  summary,
  loading,
  onOpenSection
}: KGMaintenanceOverviewProps) {
  if (loading && !summary) {
    return (
      <div className="space-y-3">
        <div className="bg-muted h-20 animate-pulse rounded-lg" />
        <div className="grid gap-3 md:grid-cols-3">
          <div className="bg-muted h-24 animate-pulse rounded-lg" />
          <div className="bg-muted h-24 animate-pulse rounded-lg" />
          <div className="bg-muted h-24 animate-pulse rounded-lg" />
        </div>
      </div>
    )
  }

  if (!summary) {
    return (
      <div className="border-border/70 bg-muted/20 rounded-lg border p-6">
        <h2 className="text-sm font-semibold">Run KB iteration review first</h2>
        <p className="text-muted-foreground mt-2 max-w-[64ch] text-sm">
          No review package is available for this workspace yet.
        </p>
      </div>
    )
  }

  const quality = summary.quality?.overall ?? 0
  const evidenceCoverage = getEvidenceCoveragePercent(summary.quality)

  return (
    <div className="space-y-4">
      <div className="border-border/70 bg-muted/20 rounded-lg border p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold">{summary.workspace}</h2>
            <p className="text-muted-foreground mt-1 text-sm">
              {formatRunSubtitle(summary.profile, summary.phase)}
            </p>
          </div>
          <div className="text-muted-foreground text-right text-xs">
            <div>Run: {summary.latestRunId}</div>
            <div>{summary.generatedAt || 'No timestamp'}</div>
          </div>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-6">
        <button type="button" className={metricClass} onClick={() => onOpenSection('check')}>
          <div className="text-muted-foreground text-xs">Nodes</div>
          <div className="mt-1 text-2xl font-semibold">{summary.counts.nodes}</div>
        </button>
        <button type="button" className={metricClass} onClick={() => onOpenSection('check')}>
          <div className="text-muted-foreground text-xs">Relations</div>
          <div className="mt-1 text-2xl font-semibold">{summary.counts.edges}</div>
        </button>
        <button type="button" className={metricClass} onClick={() => onOpenSection('check')}>
          <div className="text-muted-foreground text-xs">Sources</div>
          <div className="mt-1 text-2xl font-semibold">{summary.counts.sources}</div>
        </button>
        <button type="button" className={metricClass} onClick={() => onOpenSection('check')}>
          <div className="text-muted-foreground text-xs">Quality</div>
          <div className="mt-1 text-2xl font-semibold">{quality}</div>
        </button>
        <button type="button" className={metricClass} onClick={() => onOpenSection('approval')}>
          <div className="text-muted-foreground text-xs">Pending</div>
          <div className="mt-1 text-2xl font-semibold">{summary.pendingApprovalCount}</div>
        </button>
        <button type="button" className={metricClass} onClick={() => onOpenSection('check')}>
          <div className="text-muted-foreground text-xs">High Risk</div>
          <div className="mt-1 text-2xl font-semibold">{summary.highRiskFindingCount}</div>
        </button>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_280px]">
        <section className="border-border/70 rounded-lg border p-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h3 className="text-sm font-semibold">Quality Subscores</h3>
            <span className="text-muted-foreground text-xs">Overall {quality}/100</span>
          </div>
          <div className="space-y-3">
            {Object.entries(summary.quality?.subscores || {}).map(([key, value]) => (
              <div key={key}>
                <div className="mb-1 flex justify-between text-xs">
                  <span className="text-muted-foreground">{key}</span>
                  <span>{value}</span>
                </div>
                <Progress value={value} className="h-2" />
              </div>
            ))}
          </div>
        </section>

        <section className="border-border/70 rounded-lg border p-4">
          <h3 className="text-sm font-semibold">Evidence Coverage</h3>
          <div className="mt-3 text-3xl font-semibold">{evidenceCoverage}%</div>
          <Progress value={evidenceCoverage} className="mt-3 h-2" />
          <button
            type="button"
            className="text-muted-foreground hover:text-foreground mt-4 text-sm"
            onClick={() => onOpenSection('check')}
          >
            Open evidence review
          </button>
        </section>
      </div>
    </div>
  )
}
