import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Database } from 'lucide-react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import { useSettingsStore } from '@/stores/settings'
import { getKnowledgeBases, getPopularLabels } from '@/api/lightrag'
import { popularLabelsDefaultLimit } from '@/lib/constants'

/** Special workspace value that means "aggregate all knowledge bases". */
const ALL_WORKSPACES = '*'

/**
 * Knowledge-base (workspace) selector for the knowledge graph page.
 *
 * The graph endpoints are workspace-scoped: each knowledge base keeps its own
 * isolated graph storage.  An "All Knowledge Bases" option (value ``"*"``)
 * aggregates graph data from every discovered KB so the user can see the
 * combined graph in one view.
 */
const GraphWorkspaceSelect = () => {
  const { t } = useTranslation()
  const graphWorkspace = useSettingsStore.use.graphWorkspace()
  const [workspaces, setWorkspaces] = useState<string[]>([])
  const [names, setNames] = useState<Record<string, string>>({})
  const [loaded, setLoaded] = useState(false)

  // Load knowledge bases once on mount; auto-switch when the primary
  // workspace is empty but a knowledge base has graph data.
  useEffect(() => {
    let cancelled = false

    const run = async () => {
      try {
        const kbs = await getKnowledgeBases()
        if (cancelled) return
        const ids = kbs.map((kb) => kb.id)
        setWorkspaces(ids)
        setNames(
          Object.fromEntries(kbs.map((kb) => [kb.id, kb.name || kb.id]))
        )

        // Auto-switch only when still on the empty default workspace.
        if (useSettingsStore.getState().graphWorkspace === 'default' && kbs.length > 0) {
          const popular = await getPopularLabels(popularLabelsDefaultLimit, 'default')
          if (!cancelled && popular.length === 0) {
            useSettingsStore.getState().setGraphWorkspace(ALL_WORKSPACES)
          }
        }
      } catch (error) {
        console.error('Failed to load knowledge bases:', error)
      } finally {
        if (!cancelled) setLoaded(true)
      }
    }

    void run()
    return () => {
      cancelled = true
    }
  }, [])

  const handleChange = useCallback(
    (value: string) => {
      if (value === graphWorkspace) return
      useSettingsStore.getState().setGraphWorkspace(value)
    },
    [graphWorkspace]
  )

  // Valid selections: '*' (all), plus each discovered KB id.
  const validIds = [ALL_WORKSPACES, ...workspaces]
  const selected = validIds.includes(graphWorkspace) ? graphWorkspace : ''

  return (
    <div className="mr-1 flex items-center">
      <Select value={selected} onValueChange={handleChange} disabled={!loaded}>
        <SelectTrigger className="h-8 w-auto min-w-[8rem] max-w-[14rem] gap-1 px-2 text-xs">
          <Database className="h-3.5 w-3.5 opacity-60" />
          <SelectValue placeholder={t('graphPanel.workspaceSelect.placeholder')} />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={ALL_WORKSPACES}>
            {t('graphPanel.workspaceSelect.all', 'All Knowledge Bases')}
          </SelectItem>
          {workspaces.map((id) => (
            <SelectItem key={id} value={id}>
              {names[id] || id}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

export default GraphWorkspaceSelect
