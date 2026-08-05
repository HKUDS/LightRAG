import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  DatabaseIcon,
  FileTextIcon,
  CheckCircle2Icon,
  LoaderIcon,
  AlertCircleIcon,
  ClockIcon,
  RefreshCwIcon,
  NetworkIcon,
  LinkIcon,
  LayersIcon,
  ServerIcon,
  CpuIcon,
  Trash2Icon
} from 'lucide-react'

import { cn } from '@/lib/utils'
import Button from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import StatCard, { type StatTone } from '@/components/dashboard/StatCard'
import { useSettingsStore } from '@/stores/settings'
import { useBackendState } from '@/stores/state'
import {
  getDashboardStats,
  getSystemInfo,
  sweepOrphans,
  type DashboardStatsResponse,
  type SystemInfoResponse
} from '@/api/lightrag'
import { toast } from 'sonner'
import { errorMessage } from '@/lib/utils'

/** Auto-refresh cadence for dashboard metrics. */
const REFRESH_INTERVAL_MS = 30_000

interface StatusBarDatum {
  key: string
  label: string
  value: number
  className: string
}

/**
 * Horizontal stacked bar summarising document status distribution.
 *
 * Rendered as a labelled list + bar rather than a charting-library dependency,
 * which keeps the bundle small and stays fully accessible.
 */
function StatusDistribution({
  data,
  total,
  loading
}: {
  data: StatusBarDatum[]
  total: number
  loading: boolean
}) {
  const { t } = useTranslation()

  if (loading) {
    return <div className="bg-muted/40 h-2.5 w-full animate-pulse rounded-full" />
  }

  if (total === 0) {
    return (
      <p className="text-muted-foreground py-6 text-center text-sm">
        {t('dashboard.noDocuments', 'No documents yet')}
      </p>
    )
  }

  return (
    <div className="space-y-4">
      <div
        className="bg-muted/30 flex h-2.5 w-full overflow-hidden rounded-full"
        role="img"
        aria-label={t('dashboard.statusDistribution', 'Document status distribution')}
      >
        {data
          .filter((d) => d.value > 0)
          .map((d) => (
            <div
              key={d.key}
              className={cn('h-full transition-all duration-500', d.className)}
              style={{ width: `${(d.value / total) * 100}%` }}
              title={`${d.label}: ${d.value}`}
            />
          ))}
      </div>

      <ul className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-4">
        {data.map((d) => (
          <li key={d.key} className="flex items-center gap-2 text-xs">
            <span className={cn('size-2 shrink-0 rounded-full', d.className)} aria-hidden="true" />
            <span className="text-muted-foreground truncate">{d.label}</span>
            <span className="ml-auto font-medium tabular-nums">{d.value.toLocaleString()}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/** A single label/value row in the system configuration panel. */
function ConfigRow({ label, value }: { label: string; value?: string | number | null }) {
  if (value === null || value === undefined || value === '') return null
  return (
    <div className="flex items-start justify-between gap-4 py-2">
      <span className="text-muted-foreground shrink-0 text-xs">{label}</span>
      <span className="truncate text-right text-xs font-medium" title={String(value)}>
        {String(value)}
      </span>
    </div>
  )
}

export default function DashboardView() {
  const { t } = useTranslation()
  const setCurrentTab = useSettingsStore.use.setCurrentTab()
  const health = useBackendState.use.health()
  const pipelineBusy = useBackendState.use.pipelineBusy()

  const [stats, setStats] = useState<DashboardStatsResponse | null>(null)
  const [system, setSystem] = useState<SystemInfoResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [sweeping, setSweeping] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const prevBusyRef = useRef(pipelineBusy)

  const fetchData = useCallback(async (showSpinner = false) => {
    if (showSpinner) setLoading(true)
    try {
      const [statsData, systemData] = await Promise.all([getDashboardStats(), getSystemInfo()])
      setStats(statsData)
      setSystem(systemData)
      setError(null)
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setLoading(false)
    }
  }, [])

  const handleSweepOrphans = useCallback(async () => {
    setSweeping(true)
    try {
      const result = await sweepOrphans()
      const total = result.total_entities_removed + result.total_relations_removed
      if (total > 0) {
        toast.success(
          `Removed ${result.total_entities_removed} orphan entities and ${result.total_relations_removed} orphan relations across ${result.workspaces.length} workspace(s).`
        )
      } else {
        toast.info('No orphan entities or relations found.')
      }
      await fetchData()
    } catch (err) {
      toast.error(errorMessage(err) || 'Failed to sweep orphans')
    } finally {
      setSweeping(false)
    }
  }, [fetchData])

  // Initial load + polling. The interval is cleared on unmount so navigating
  // away from the dashboard stops the background traffic.
  useEffect(() => {
    let cancelled = false

    const run = async () => {
      if (cancelled) return
      await fetchData()
    }

    run()
    const timer = setInterval(run, REFRESH_INTERVAL_MS)

    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [fetchData])

  // When the pipeline transitions from busy → idle (e.g. after a document
  // deletion or scan completes), immediately refresh the stats so the
  // dashboard reflects the new state without waiting for the next poll.
  useEffect(() => {
    const wasBusy = prevBusyRef.current
    prevBusyRef.current = pipelineBusy
    if (wasBusy && !pipelineBusy) {
      fetchData()
    }
  }, [pipelineBusy, fetchData])

  const statusData = useMemo<StatusBarDatum[]>(() => {
    if (!stats) return []
    return [
      {
        key: 'processed',
        label: t('dashboard.processed', 'Completed'),
        value: stats.processed_count,
        className: 'bg-emerald-500'
      },
      {
        key: 'processing',
        label: t('dashboard.processing', 'Processing'),
        value: stats.processing_count,
        className: 'bg-blue-500'
      },
      {
        key: 'pending',
        label: t('dashboard.pending', 'Pending'),
        value: stats.pending_count,
        className: 'bg-amber-500'
      },
      {
        key: 'failed',
        label: t('dashboard.failed', 'Failed'),
        value: stats.failed_count,
        className: 'bg-rose-500'
      }
    ]
  }, [stats, t])

  const primaryCards = useMemo(
    () => [
      {
        key: 'kb',
        label: t('dashboard.knowledgeBases', 'Knowledge Bases'),
        value: stats?.knowledge_base_count ?? 0,
        icon: DatabaseIcon,
        tone: 'brand' as StatTone,
        onClick: () => setCurrentTab('knowledge-base')
      },
      {
        key: 'docs',
        label: t('dashboard.documents', 'Documents'),
        value: stats?.document_count ?? 0,
        icon: FileTextIcon,
        tone: 'info' as StatTone,
        onClick: () => setCurrentTab('documents')
      },
      {
        key: 'processed',
        label: t('dashboard.processed', 'Completed'),
        value: stats?.processed_count ?? 0,
        icon: CheckCircle2Icon,
        tone: 'success' as StatTone
      },
      {
        key: 'processing',
        label: t('dashboard.processing', 'Processing'),
        value: stats?.processing_count ?? 0,
        icon: LoaderIcon,
        tone: 'warning' as StatTone,
        hint: stats?.pipeline_busy
          ? t('dashboard.pipelineBusy', 'Pipeline running')
          : undefined
      },
      {
        key: 'failed',
        label: t('dashboard.failed', 'Failed'),
        value: stats?.failed_count ?? 0,
        icon: AlertCircleIcon,
        tone: 'danger' as StatTone
      }
    ],
    [stats, t, setCurrentTab]
  )

  const graphCards = useMemo(
    () => [
      {
        key: 'entities',
        label: t('dashboard.entities', 'Entities'),
        value: stats?.entity_count ?? 0,
        icon: NetworkIcon,
        tone: 'brand' as StatTone,
        onClick: () => setCurrentTab('knowledge-graph')
      },
      {
        key: 'relations',
        label: t('dashboard.relations', 'Relations'),
        value: stats?.relation_count ?? 0,
        icon: LinkIcon,
        tone: 'info' as StatTone,
        onClick: () => setCurrentTab('knowledge-graph')
      },
      {
        key: 'chunks',
        label: t('dashboard.chunks', 'Text Chunks'),
        value: stats?.chunk_count ?? 0,
        icon: LayersIcon,
        tone: 'neutral' as StatTone
      }
    ],
    [stats, t, setCurrentTab]
  )

  const lastUpdated = stats?.generated_at
    ? new Date(stats.generated_at * 1000).toLocaleTimeString()
    : null

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1600px] space-y-6 p-6">
        {/* Page header */}
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div className="min-w-0">
            <h1 className="text-2xl font-semibold tracking-tight">
              {t('dashboard.title', 'Dashboard')}
            </h1>
            <p className="text-muted-foreground mt-1 text-sm">
              {t('dashboard.subtitle', 'Overview of your knowledge base and system status')}
            </p>
          </div>

          <div className="flex items-center gap-3">
            {lastUpdated && (
              <span className="text-muted-foreground hidden items-center gap-1.5 text-xs sm:flex">
                <ClockIcon className="size-3.5" aria-hidden="true" />
                {t('dashboard.lastUpdated', 'Updated')} {lastUpdated}
              </span>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={() => fetchData(true)}
              disabled={loading}
              className="gap-2"
            >
              <RefreshCwIcon
                className={cn('size-3.5', loading && 'animate-spin')}
                aria-hidden="true"
              />
              {t('dashboard.refresh', 'Refresh')}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => void handleSweepOrphans()}
              disabled={sweeping || pipelineBusy}
              className="gap-2"
            >
              <Trash2Icon
                className={cn('size-3.5', sweeping && 'animate-spin')}
                aria-hidden="true"
              />
              {t('dashboard.sweepOrphans', 'Sweep Orphans')}
            </Button>
          </div>
        </header>

        {error && (
          <div
            role="alert"
            className="flex items-center gap-2 rounded-lg bg-rose-500/10 px-4 py-3 text-sm text-rose-400 ring-1 ring-rose-500/20 ring-inset"
          >
            <AlertCircleIcon className="size-4 shrink-0" aria-hidden="true" />
            <span className="truncate">{error}</span>
          </div>
        )}

        {/* Primary metrics */}
        <section aria-label={t('dashboard.overview', 'Overview')}>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
            {primaryCards.map((card) => (
              <StatCard
                key={card.key}
                label={card.label}
                value={card.value}
                icon={card.icon}
                tone={card.tone}
                hint={card.hint}
                loading={loading && !stats}
                onClick={card.onClick}
              />
            ))}
          </div>
        </section>

        {/* Graph metrics */}
        <section aria-label={t('dashboard.graphMetrics', 'Graph metrics')}>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            {graphCards.map((card) => (
              <StatCard
                key={card.key}
                label={card.label}
                value={card.value}
                icon={card.icon}
                tone={card.tone}
                loading={loading && !stats}
                onClick={card.onClick}
              />
            ))}
          </div>
        </section>

        {/* Bento: status distribution + system config */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <Card variant="glass" className="glass-sheen lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <LayersIcon className="text-primary size-4" aria-hidden="true" />
                {t('dashboard.statusDistribution', 'Document Status Distribution')}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <StatusDistribution
                data={statusData}
                total={stats?.document_count ?? 0}
                loading={loading && !stats}
              />
            </CardContent>
          </Card>

          <Card variant="glass" className="glass-sheen">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <ServerIcon className="text-primary size-4" aria-hidden="true" />
                {t('dashboard.systemConfig', 'System Configuration')}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {loading && !system ? (
                <div className="space-y-2">
                  {Array.from({ length: 6 }).map((_, i) => (
                    <div key={i} className="bg-muted/40 h-6 w-full animate-pulse rounded" />
                  ))}
                </div>
              ) : (
                <div className="divide-border/50 divide-y">
                  <ConfigRow
                    label={t('dashboard.status', 'Status')}
                    value={
                      health
                        ? t('dashboard.healthy', 'Healthy')
                        : t('dashboard.unhealthy', 'Unavailable')
                    }
                  />
                  <ConfigRow
                    label={t('dashboard.workspace', 'Workspace')}
                    value={system?.workspace || t('dashboard.defaultWorkspace', 'default')}
                  />
                  <ConfigRow label={t('dashboard.llmModel', 'LLM')} value={system?.llm_model} />
                  <ConfigRow
                    label={t('dashboard.embeddingModel', 'Embedding')}
                    value={system?.embedding_model}
                  />
                  <ConfigRow
                    label={t('dashboard.graphStorage', 'Graph Storage')}
                    value={system?.graph_storage}
                  />
                  <ConfigRow
                    label={t('dashboard.vectorStorage', 'Vector Storage')}
                    value={system?.vector_storage}
                  />
                  <ConfigRow
                    label={t('dashboard.chunkSize', 'Chunk Size')}
                    value={system?.chunk_size}
                  />
                  <ConfigRow
                    label={t('dashboard.version', 'Version')}
                    value={
                      system?.core_version && system?.api_version
                        ? `${system.core_version} / ${system.api_version}`
                        : null
                    }
                  />
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Quick actions */}
        <section aria-label={t('dashboard.quickActions', 'Quick actions')}>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            {[
              {
                key: 'documents',
                icon: DatabaseIcon,
                title: t('dashboard.manageDocuments', 'Manage Knowledge Base'),
                desc: t('dashboard.manageDocumentsDesc', 'Upload and track documents'),
                tab: 'knowledge-base' as const
              },
              {
                key: 'retrieval',
                icon: CpuIcon,
                title: t('dashboard.runQuery', 'Run a Query'),
                desc: t('dashboard.runQueryDesc', 'Ask questions across your data'),
                tab: 'retrieval' as const
              },
              {
                key: 'graph',
                icon: NetworkIcon,
                title: t('dashboard.exploreGraph', 'Explore Graph'),
                desc: t('dashboard.exploreGraphDesc', 'Visualise entities and relations'),
                tab: 'knowledge-graph' as const
              }
            ].map((action) => (
              <Card
                key={action.key}
                variant="glass"
                interactive
                className="glass-sheen p-4"
                onClick={() => setCurrentTab(action.tab)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    setCurrentTab(action.tab)
                  }
                }}
              >
                <div className="flex items-center gap-3">
                  <span className="bg-primary/12 text-primary ring-primary/20 flex size-10 shrink-0 items-center justify-center rounded-lg ring-1 ring-inset">
                    <action.icon className="size-5" aria-hidden="true" />
                  </span>
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{action.title}</p>
                    <p className="text-muted-foreground truncate text-xs">{action.desc}</p>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}
