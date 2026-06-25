import { useSettingsStore } from '@/stores/settings'
import { useGraphStore } from '@/stores/graph'
import { useTranslation } from 'react-i18next'

/**
 * Component that displays current values of important graph settings
 * Positioned to the right of the toolbar at the bottom-left corner
 */
const SettingsDisplay = () => {
  const { t } = useTranslation()
  const graphQueryMaxDepth = useSettingsStore.use.graphQueryMaxDepth()
  // Live counts of the rendered graph (reactive: updated on build/expand/prune).
  // Reading sigmaGraph.order/.size directly would not re-render, since
  // expand/prune mutate the graph in place.
  const graphNodeCount = useGraphStore.use.graphNodeCount()
  const graphEdgeCount = useGraphStore.use.graphEdgeCount()

  return (
    <div className="absolute bottom-4 left-[calc(1rem+2.5rem)] flex items-center gap-2 text-xs text-gray-400">
      <div>{t('graphPanel.sideBar.settings.depth')}: {graphQueryMaxDepth}</div>
      <div>{t('graphPanel.sideBar.settings.node')}: {graphNodeCount}</div>
      <div>{t('graphPanel.sideBar.settings.edge')}: {graphEdgeCount}</div>
    </div>
  )
}

export default SettingsDisplay
