import { useCallback } from 'react'
import { AsyncSelect } from '@/components/ui/AsyncSelect'
import { useSettingsStore } from '@/stores/settings'
import { useGraphStore } from '@/stores/graph'
import { labelListLimit, controlButtonVariant } from '@/lib/constants'
import MiniSearch from 'minisearch'
import { useTranslation } from 'react-i18next'
import { RefreshCw } from 'lucide-react'
import Button from '@/components/ui/Button'

const GraphLabels = () => {
  const { t } = useTranslation()
  const label = useSettingsStore.use.queryLabel()
  const allDatabaseLabels = useGraphStore.use.allDatabaseLabels()

  // Remove initial label fetch effect as it's now handled by fetchGraph based on lastSuccessfulQueryLabel

  const getSearchEngine = useCallback(() => {
    // Create search engine
    const searchEngine = new MiniSearch({
      idField: 'id',
      fields: ['value'],
      searchOptions: {
        prefix: true,
        fuzzy: 0.2,
        boost: {
          label: 2
        }
      }
    })

    // Add documents
    const documents = allDatabaseLabels.map((str, index) => ({ id: index, value: str }))
    searchEngine.addAll(documents)

    return {
      labels: allDatabaseLabels,
      searchEngine
    }
  }, [allDatabaseLabels])

  const fetchData = useCallback(
    async (query?: string): Promise<string[]> => {
      const { labels, searchEngine } = getSearchEngine()

      let result: string[] = labels
      if (query) {
        // Search labels
        result = searchEngine.search(query).map((r: { id: number }) => labels[r.id])
      }

      return result.length <= labelListLimit
        ? result
        : [...result.slice(0, labelListLimit), '...']
    },
    [getSearchEngine]
  )

  const handleRefresh = useCallback(() => {
    // Reset fetch status flags
    useGraphStore.getState().setLabelsFetchAttempted(false)
    useGraphStore.getState().setGraphDataFetchAttempted(false)

    // Clear last successful query label to ensure labels are fetched
    useGraphStore.getState().setLastSuccessfulQueryLabel('')

    // Get current label
    const currentLabel = useSettingsStore.getState().queryLabel

    // If current label is empty, use default label '*'
    if (!currentLabel) {
      useSettingsStore.getState().setQueryLabel('*')
    } else {
      // Trigger data reload
      useSettingsStore.getState().setQueryLabel('')
      setTimeout(() => {
        useSettingsStore.getState().setQueryLabel(currentLabel)
      }, 0)
    }
  }, []);

  return (
    <div className="flex items-center">
      {/* Always show refresh button */}
      <Button
        size="icon"
        variant={controlButtonVariant}
        onClick={handleRefresh}
        tooltip={t('graphPanel.graphLabels.refreshTooltip')}
        className="mr-1"
      >
        <RefreshCw className="h-4 w-4" />
      </Button>
      <AsyncSelect<string>
        className="ml-2"
        triggerClassName="max-h-8"
        searchInputClassName="max-h-8"
        triggerTooltip={t('graphPanel.graphLabels.selectTooltip')}
        fetcher={fetchData}
        renderOption={(item) => <div>{item}</div>}
        getOptionValue={(item) => item}
        getDisplayValue={(item) => <div>{item}</div>}
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

          // Reset graphDataFetchAttempted flag to ensure data fetch is triggered
          useGraphStore.getState().setGraphDataFetchAttempted(false);

          // Update the label to trigger data loading
          useSettingsStore.getState().setQueryLabel(newLabel);
        }}
        clearable={false}  // Prevent clearing value on reselect
      />
    </div>
  )
}

export default GraphLabels
