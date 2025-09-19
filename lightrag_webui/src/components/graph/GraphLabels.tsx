import { useCallback, useEffect } from 'react'
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
    []
  )

  const handleRefresh = useCallback(async () => {
    // Clear search history
    SearchHistoryManager.clearHistory()

    // Reinitialize with popular labels
    try {
      const popularLabels = await getPopularLabels(popularLabelsDefaultLimit)
      await SearchHistoryManager.initializeWithDefaults(popularLabels)
    } catch (error) {
      console.error('Failed to reload popular labels:', error)
      // No fallback needed
    }

    // Reset fetch status flags to trigger UI refresh
    useGraphStore.getState().setLabelsFetchAttempted(false)
    useGraphStore.getState().setGraphDataFetchAttempted(false)

    // Clear last successful query label to ensure labels are fetched,
    // which is the key to forcing a data refresh.
    useGraphStore.getState().setLastSuccessfulQueryLabel('')

    // Reset to default label to ensure consistency
    useSettingsStore.getState().setQueryLabel('*')

    // Force a data refresh by incrementing the version counter in the graph store.
    // This is the reliable way to trigger a re-fetch of the graph data.
    useGraphStore.getState().incrementGraphDataVersion()
  }, []);

  return (
    <div className="flex items-center">
      {/* Always show refresh button */}
      <Button
        size="icon"
        variant={controlButtonVariant}
        onClick={handleRefresh}
        tooltip={t('graphPanel.graphLabels.refreshTooltip')}
        className="mr-2"
      >
        <RefreshCw className="h-4 w-4" />
      </Button>
      <AsyncSelect<string>
        className="min-w-[300px]"
        triggerClassName="max-h-8"
        searchInputClassName="max-h-8"
        triggerTooltip={t('graphPanel.graphLabels.selectTooltip')}
        fetcher={fetchData}
        renderOption={(item) => <div style={{ whiteSpace: 'pre' }}>{item}</div>}
        getOptionValue={(item) => item}
        getDisplayValue={(item) => <div style={{ whiteSpace: 'pre' }}>{item}</div>}
        notFound={<div className="py-6 text-center text-sm">No labels found</div>}
        label={t('graphPanel.graphLabels.label')}
        placeholder={t('graphPanel.graphLabels.placeholder')}
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
