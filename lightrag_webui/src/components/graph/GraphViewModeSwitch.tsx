import { Database, Stethoscope } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import Button from '@/components/ui/Button'
import type { GraphViewMode } from '@/api/lightrag'
import { cn } from '@/lib/utils'
import { useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'

const resolveGraphViewMode = (mode: unknown): GraphViewMode => (mode === 'raw' ? 'raw' : 'medical')

const GraphViewModeSwitch = () => {
  const { t } = useTranslation()
  const graphViewMode = useSettingsStore((state) =>
    resolveGraphViewMode((state as { graphViewMode?: unknown }).graphViewMode)
  )

  const setMode = (nextMode: GraphViewMode) => {
    if (nextMode === graphViewMode) {
      return
    }

    const settingsState = useSettingsStore.getState() as {
      setGraphViewMode?: (mode: GraphViewMode) => void
    }

    if (settingsState.setGraphViewMode) {
      settingsState.setGraphViewMode(nextMode)
    } else {
      ;(useSettingsStore as any).setState({ graphViewMode: nextMode })
    }

    const graphState = useGraphStore.getState()
    graphState.clearSelection()
    graphState.setGraphDataFetchAttempted(false)
    graphState.setLastSuccessfulQueryLabel('')
    graphState.incrementGraphDataVersion()
  }

  const choices: Array<{
    mode: GraphViewMode
    label: string
    tooltip: string
    icon: typeof Stethoscope
  }> = [
    {
      mode: 'medical',
      label: t('graphPanel.viewMode.medical'),
      tooltip: t('graphPanel.viewMode.medicalTooltip'),
      icon: Stethoscope
    },
    {
      mode: 'raw',
      label: t('graphPanel.viewMode.raw'),
      tooltip: t('graphPanel.viewMode.rawTooltip'),
      icon: Database
    }
  ]

  return (
    <div className="bg-background/80 flex h-8 items-center rounded-md border p-0.5 shadow-sm backdrop-blur">
      {choices.map(({ mode, label, tooltip, icon: Icon }) => {
        const active = graphViewMode === mode

        return (
          <Button
            key={mode}
            type="button"
            variant={active ? 'secondary' : 'ghost'}
            size="sm"
            aria-pressed={active}
            onClick={() => setMode(mode)}
            tooltip={tooltip}
            className={cn(
              'h-7 rounded px-2 text-xs shadow-none',
              active && 'bg-primary text-primary-foreground hover:bg-primary/90 hover:text-primary-foreground'
            )}
          >
            <Icon className="h-3.5 w-3.5" />
            <span>{label}</span>
          </Button>
        )
      })}
    </div>
  )
}

export default GraphViewModeSwitch
