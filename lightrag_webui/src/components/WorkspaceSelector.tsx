import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useSettingsStore } from '@/stores/settings'
import { getWorkspaces, type Workspace } from '@/api/lightrag'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'
import { cn } from '@/lib/utils'

const REFRESH_INTERVAL_MS = 30_000

interface WorkspaceSelectorProps {
  className?: string
}

export default function WorkspaceSelector({ className }: WorkspaceSelectorProps) {
  const { t } = useTranslation()
  const currentWorkspace = useSettingsStore.use.currentWorkspace()
  const setCurrentWorkspace = useSettingsStore.use.setCurrentWorkspace()
  const [workspaces, setWorkspaces] = useState<Workspace[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const fetchWorkspaces = async () => {
    setIsLoading(true)
    try {
      const data = await getWorkspaces()
      setWorkspaces(data)
      // Check if current workspace is still in the list
      const current = useSettingsStore.getState().currentWorkspace
      if (current && !data.some((w: { name: string }) => w.name === current)) {
        useSettingsStore.getState().setCurrentWorkspace(null)
      }
    } catch {
      // Graceful degradation: keep empty list, show only "None" option
      setWorkspaces([])
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchWorkspaces()
    const interval = setInterval(fetchWorkspaces, REFRESH_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [])

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value
    setCurrentWorkspace(value === '' ? null : value)
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <select
            value={currentWorkspace ?? ''}
            onChange={handleChange}
            disabled={isLoading}
            className={cn('h-6 max-w-[120px] rounded border border-input bg-background px-1.5 text-xs shadow-sm transition-colors hover:border-ring focus:border-ring focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50', className)}
            aria-label={t('workspace.selector', 'Workspace')}
          >
            <option value="">{t('workspace.none', 'None')}</option>
            {workspaces.map((ws) => (
              <option key={ws.name} value={ws.name}>
                {ws.name}
              </option>
            ))}
          </select>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          {t('workspace.selector', 'Workspace')}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}
