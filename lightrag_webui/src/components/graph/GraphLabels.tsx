import { useCallback, useEffect, useState } from 'react'
import { AsyncSelect } from '@/components/ui/AsyncSelect'
import { useSettingsStore } from '@/stores/settings'
import { useGraphStore } from '@/stores/graph'
import {
  dropdownDisplayLimit,
  controlButtonVariant,
  popularLabelsDefaultLimit,
  searchLabelsDefaultLimit
} from '@/lib/constants'
import { useTranslation } from 'react-i18next'
import { RefreshCw } from 'lucide-react'
import Button from '@/components/ui/Button'
import { SearchHistoryManager } from '@/utils/SearchHistoryManager'
import { getPopularLabels, searchLabels } from '@/api/lightrag'

const GraphLabels = () => {
  const { t } = useTranslation()
  const label = useSettingsStore.use.queryLabel()
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [refreshTrigger, setRefreshTrigger] = useState(0)
  const [selectKey, setSelectKey] = useState(0)

  // Dynamic tooltip based on current label state
  const getRefreshTooltip = useCallback(() => {
    if (isRefreshing) {
      return t('graphPanel.graphLabels.refreshingTooltip')
    }

    if (!label || label === '*') {
      return t('graphPanel.graphLabels.refreshGlobalTooltip')
    } else {
      return t('graphPanel.graphLabels.refreshCurrentLabelTooltip', { label })
    }
  }, [label, t, isRefreshing])

  // Initialize search history on component mount
  useEffect(() => {
    const initializeHistory = async () => {
      const history = SearchHistoryManager.getHistory()

      if (history.length === 0) {
        // If no history exists, fetch popular labels and initialize
        try {
          const popularLabels = await getPopularLabels(popularLabelsDefaultLimit)
          await SearchHistoryManager.initializeWithDefaults(popularLabels)
        } catch (error) {
          console.error('Failed to initialize search history:', error)
          // No fallback needed, API is the source of truth
        }
      }
    }

    initializeHistory()
  }, [])

  const fetchData = useCallback(
    async (query?: string): Promise<string[]> => {
      let results: string[] = [];
      if (!query || query.trim() === '' || query.trim() === '*') {
        // Empty query: return search history
        results = SearchHistoryManager.getHistoryLabels(dropdownDisplayLimit)
      } else {
        // Non-empty query: call backend search API
        try {
          const apiResults = await searchLabels(query.trim(), searchLabelsDefaultLimit)
          results = apiResults.length <= dropdownDisplayLimit
            ? apiResults
            : [...apiResults.slice(0, dropdownDisplayLimit), '...']
        } catch (error) {
          console.error('Search API failed, falling back to local history search:', error)

          // Fallback to local history search
          const history = SearchHistoryManager.getHistory()
          const queryLower = query.toLowerCase().trim()
          results = history
            .filter(item => item.label.toLowerCase().includes(queryLower))
            .map(item => item.label)
            .slice(0, dropdownDisplayLimit)
        }
      }
      // Always show '*' at the top, and remove duplicates
      const finalResults = ['*', ...results.filter(label => label !== '*')];
      return finalResults;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [refreshTrigger] // Intentionally added to trigger re-creation when data changes
  )

  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true)

    // Clear legend cache to ensure legend is re-generated on refresh
    useGraphStore.getState().setTypeColorMap(new Map<string, string>())

    try {
      let currentLabel = label

      // If queryLabel is empty, set it to '*'
      if (!currentLabel || currentLabel.trim() === '') {
        useSettingsStore.getState().setQueryLabel('*')
        currentLabel = '*'
      }

      if (currentLabel && currentLabel !== '*') {
        // Scenario 1: Has specific label, try to refresh current label
        console.log(`Refreshing current label: ${currentLabel}`)

        // Reset graph data fetch status to trigger refresh
        useGraphStore.getState().setGraphDataFetchAttempted(false)
        useGraphStore.getState().setLastSuccessfulQueryLabel('')

        // Force data refresh for current label
        useGraphStore.getState().incrementGraphDataVersion()

        // Note: If the current label has no data after refresh,
        // the fallback logic would be handled by the graph component itself
        // For now, we keep the current label and let the user see the result

      } else {
        // Scenario 3: queryLabel is "*", refresh global data and popular labels
        console.log('Refreshing global data and popular labels')

        try {
          // Re-fetch popular labels and update search history
          const popularLabels = await getPopularLabels(popularLabelsDefaultLimit)
          SearchHistoryManager.clearHistory()

          if (popularLabels.length === 0) {
            // If no popular labels, provide fallback defaults
            const fallbackLabels = ['entity', 'relationship', 'document', 'concept']
            await SearchHistoryManager.initializeWithDefaults(fallbackLabels)
          } else {
            await SearchHistoryManager.initializeWithDefaults(popularLabels)
          }
        } catch (error) {
          console.error('Failed to reload popular labels:', error)
          // Provide fallback even if API fails
          const fallbackLabels = ['entity', 'relationship', 'document']
          SearchHistoryManager.clearHistory()
          await SearchHistoryManager.initializeWithDefaults(fallbackLabels)
        }

        // Reset graph data fetch status
        useGraphStore.getState().setGraphDataFetchAttempted(false)
        useGraphStore.getState().setLastSuccessfulQueryLabel('')

        // Force global data refresh
        useGraphStore.getState().incrementGraphDataVersion()

        // Ensure data update completes before triggering UI refresh
        await new Promise(resolve => setTimeout(resolve, 0))

        // Trigger both refresh mechanisms to ensure dropdown updates
        setRefreshTrigger(prev => prev + 1)
        setSelectKey(prev => prev + 1)
      }
    } catch (error) {
      console.error('Error during refresh:', error)
    } finally {
      setIsRefreshing(false)
    }
  }, [label])

  return (
    <div className="flex items-center">
      {/* Always show refresh button */}
      <Button
        size="icon"
        variant={controlButtonVariant}
        onClick={handleRefresh}
        tooltip={getRefreshTooltip()}
        className="mr-2"
        disabled={isRefreshing}
      >
        <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
      </Button>
      <AsyncSelect<string>
        key={selectKey} // Force re-render when data changes
        className="min-w-[300px]"
        triggerClassName="max-h-8"
        searchInputClassName="max-h-8"
        triggerTooltip={t('graphPanel.graphLabels.selectTooltip')}
        fetcher={fetchData}
        renderOption={(item) => <div style={{ whiteSpace: 'pre' }}>{item}</div>}
        getOptionValue={(item) => item}
        getDisplayValue={(item) => <div style={{ whiteSpace: 'pre' }}>{item}</div>}
        notFound={<div className="py-6 text-center text-sm">{t('graphPanel.graphLabels.noLabels')}</div>}
        ariaLabel={t('graphPanel.graphLabels.label')}
        placeholder={t('graphPanel.graphLabels.placeholder')}
        searchPlaceholder={t('graphPanel.graphLabels.placeholder')}
        noResultsMessage={t('graphPanel.graphLabels.noLabels')}
        value={label !== null ? label : '*'}
        onChange={(newLabel) => {
          const currentLabel = useSettingsStore.getState().queryLabel;

          // select the last item means query all
          if (newLabel === '...') {
            newLabel = '*';
          }

          // Handle reselecting the same label
          if (newLabel === currentLabel && newLabel !== '*') {
            newLabel = '*';
          }

          // Add selected label to search history (except for special cases)
          if (newLabel && newLabel !== '*' && newLabel !== '...' && newLabel.trim() !== '') {
            SearchHistoryManager.addToHistory(newLabel);
          }

          // Reset graphDataFetchAttempted flag to ensure data fetch is triggered
          useGraphStore.getState().setGraphDataFetchAttempted(false);

          // Update the label to trigger data loading
          useSettingsStore.getState().setQueryLabel(newLabel);
        }}
        clearable={false}  // Prevent clearing value on reselect
        debounceTime={500}
      />
    </div>
  )
}

export default GraphLabels
