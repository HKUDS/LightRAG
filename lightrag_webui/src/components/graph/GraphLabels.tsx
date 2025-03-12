import { useCallback } from 'react'
import { AsyncSelect } from '@/components/ui/AsyncSelect'
import { useSettingsStore } from '@/stores/settings'
import { useGraphStore } from '@/stores/graph'
import { labelListLimit } from '@/lib/constants'
import MiniSearch from 'minisearch'
import { useTranslation } from 'react-i18next'

const GraphLabels = () => {
  const { t } = useTranslation()
  const label = useSettingsStore.use.queryLabel()
  const graphLabels = useGraphStore.use.graphLabels()

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
    const documents = graphLabels.map((str, index) => ({ id: index, value: str }))
    searchEngine.addAll(documents)

    return {
      labels: graphLabels,
      searchEngine
    }
  }, [graphLabels])

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
        : [...result.slice(0, labelListLimit), t('graphLabels.andOthers', { count: result.length - labelListLimit })]
    },
    [getSearchEngine, t]
  )

  const setQueryLabel = useCallback((newLabel: string) => {
    if (newLabel.startsWith('And ') && newLabel.endsWith(' others')) return

    const currentLabel = useSettingsStore.getState().queryLabel

    if (newLabel === '*' && currentLabel === '*') {
      // When reselecting '*', just set it again to trigger a new fetch
      useSettingsStore.getState().setQueryLabel('*')
    } else if (newLabel === currentLabel && newLabel !== '*') {
      // When selecting the same label (except '*'), switch to '*'
      useSettingsStore.getState().setQueryLabel('*')
    } else {
      useSettingsStore.getState().setQueryLabel(newLabel)
    }
  }, [])

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
      value={label !== null ? label : ''}
      onChange={setQueryLabel}
      clearable={false}  // Prevent clearing value on reselect
    />
  )
}

export default GraphLabels
