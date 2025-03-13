import { useCallback, useEffect, useRef } from 'react'
import { AsyncSelect } from '@/components/ui/AsyncSelect'
import { useSettingsStore } from '@/stores/settings'
import { useGraphStore } from '@/stores/graph'
import { labelListLimit } from '@/lib/constants'
import MiniSearch from 'minisearch'
import { useTranslation } from 'react-i18next'

const GraphLabels = () => {
  const { t } = useTranslation()
  const label = useSettingsStore.use.queryLabel()
  const allDatabaseLabels = useGraphStore.use.allDatabaseLabels()
  const labelsLoadedRef = useRef(false)
  
  // Track if a fetch is in progress to prevent multiple simultaneous fetches
  const fetchInProgressRef = useRef(false)
  
  // Fetch labels once on component mount, using global flag to prevent duplicates
  useEffect(() => {
    // Check if we've already attempted to fetch labels in this session
    const labelsFetchAttempted = useGraphStore.getState().labelsFetchAttempted
    
    // Only fetch if we haven't attempted in this session and no fetch is in progress
    if (!labelsFetchAttempted && !fetchInProgressRef.current) {
      fetchInProgressRef.current = true
      // Set global flag to indicate we've attempted to fetch in this session
      useGraphStore.getState().setLabelsFetchAttempted(true)
      
      console.log('Fetching graph labels (once per session)...')
      
      useGraphStore.getState().fetchAllDatabaseLabels()
        .then(() => {
          labelsLoadedRef.current = true
          fetchInProgressRef.current = false
        })
        .catch((error) => {
          console.error('Failed to fetch labels:', error)
          fetchInProgressRef.current = false
          // Reset global flag to allow retry
          useGraphStore.getState().setLabelsFetchAttempted(false)
        })
    }
  }, []) // Empty dependency array ensures this only runs once on mount

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

  return (
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
        const currentLabel = useSettingsStore.getState().queryLabel

        if (newLabel === '...') {
          newLabel = '*'
        }
        if (newLabel === currentLabel && newLabel !== '*') {
          // 选择相同标签时切换到'*'
          useSettingsStore.getState().setQueryLabel('*')
        } else {
          useSettingsStore.getState().setQueryLabel(newLabel)
        }
      }}
      clearable={false}  // Prevent clearing value on reselect
    />
  )
}

export default GraphLabels
