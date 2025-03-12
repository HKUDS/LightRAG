import { useSettingsStore } from '@/stores/settings'
import { useTranslation } from 'react-i18next'

/**
 * Component that displays current values of important graph settings
 * Positioned to the right of the toolbar at the bottom-left corner
 */
const SettingsDisplay = () => {
  const { t } = useTranslation()
  const graphQueryMaxDepth = useSettingsStore.use.graphQueryMaxDepth()
  const graphMinDegree = useSettingsStore.use.graphMinDegree()

  return (
    <div className="absolute bottom-2 left-[calc(2rem+2.5rem)] flex items-center gap-2 text-xs text-gray-400">
      <div>{t('graphPanel.sideBar.settings.depth')}: {graphQueryMaxDepth}</div>
      <div>{t('graphPanel.sideBar.settings.degree')}: {graphMinDegree}</div>
    </div>
  )
}

export default SettingsDisplay
