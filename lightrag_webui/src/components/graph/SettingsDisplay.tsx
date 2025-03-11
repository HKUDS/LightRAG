import { useSettingsStore } from '@/stores/settings'

/**
 * Component that displays current values of important graph settings
 * Positioned to the right of the toolbar at the bottom-left corner
 */
const SettingsDisplay = () => {
  const graphQueryMaxDepth = useSettingsStore.use.graphQueryMaxDepth()
  const graphMinDegree = useSettingsStore.use.graphMinDegree()

  return (
    <div className="absolute bottom-2 left-[calc(2rem+2.5rem)] flex items-center gap-2 text-xs text-gray-400">
      <div>Depth: {graphQueryMaxDepth}</div>
      <div>Degree: {graphMinDegree}</div>
    </div>
  )
}

export default SettingsDisplay
