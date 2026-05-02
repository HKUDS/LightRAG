import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useSettingsStore } from '@/stores/settings'
import { getWorkspaces, type Workspace } from '@/api/lightrag'
import { FolderIcon } from 'lucide-react'

const REFRESH_INTERVAL_MS = 30_000

export default function WorkspaceSelector() {
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
    <div className="flex items-center gap-1">
      <FolderIcon className="size-3.5 text-muted-foreground" aria-hidden="true" />
      <select
        value={currentWorkspace ?? ''}
        onChange={handleChange}
        disabled={isLoading}
        className="h-6 max-w-[120px] rounded border border-input bg-background px-1.5 text-xs shadow-sm transition-colors hover:border-ring focus:border-ring focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50"
        aria-label={t('workspace.selector', 'Workspace')}
      >
        <option value="">{t('workspace.none', 'None')}</option>
        {workspaces.map((ws) => (
          <option key={ws.name} value={ws.name}>
            {ws.name}
          </option>
        ))}
      </select>
    </div>
  )
}
