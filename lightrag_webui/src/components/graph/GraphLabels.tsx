import { useCallback, useState } from 'react'
import { AsyncSelect } from '@/components/ui/AsyncSelect'
import { getGraphLabels } from '@/api/lightrag'
import { useSettingsStore } from '@/stores/settings'
import MiniSearch from 'minisearch'

const GraphLabels = () => {
  const label = useSettingsStore.use.queryLabel()
  const [labels, setLabels] = useState<{
    labels: string[]
    searchEngine: MiniSearch | null
  }>({
    labels: [],
    searchEngine: null
  })
  const [fetched, setFetched] = useState(false)

  const fetchData = useCallback(
    async (query?: string): Promise<string[]> => {
      let _labels = labels.labels
      let _searchEngine = labels.searchEngine

      if (!fetched || !_searchEngine) {
        _labels = ['*'].concat(await getGraphLabels())

        // Ensure query label exists
        if (!_labels.includes(useSettingsStore.getState().queryLabel)) {
          useSettingsStore.getState().setQueryLabel(_labels[0])
        }

        // Create search engine
        _searchEngine = new MiniSearch({
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
        const documents = _labels.map((str, index) => ({ id: index, value: str }))
        _searchEngine.addAll(documents)

        setLabels({
          labels: _labels,
          searchEngine: _searchEngine
        })
        setFetched(true)
      }
      if (!query) {
        return _labels
      }

      // Search labels
      return _searchEngine.search(query).map((result) => _labels[result.id])
    },
    [labels, fetched, setLabels, setFetched]
  )

  const setQueryLabel = useCallback((label: string) => {
    useSettingsStore.getState().setQueryLabel(label)
  }, [])

  return (
    <AsyncSelect<string>
      className="ml-2"
      triggerClassName="max-h-8"
      searchInputClassName="max-h-8"
      triggerTooltip="Select query label"
      fetcher={fetchData}
      renderOption={(item) => <div>{item}</div>}
      getOptionValue={(item) => item}
      getDisplayValue={(item) => <div>{item}</div>}
      notFound={<div className="py-6 text-center text-sm">No labels found</div>}
      label="Label"
      placeholder="Search labels..."
      value={label !== null ? label : ''}
      onChange={setQueryLabel}
    />
  )
}

export default GraphLabels
