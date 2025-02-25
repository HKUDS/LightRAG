import { useCallback } from 'react'
import { AsyncSelect } from '@/components/ui/AsyncSelect'
import { getGraphLabels } from '@/api/lightrag'
import { useSettingsStore } from '@/stores/settings'
import { useGraphStore } from '@/stores/graph'
import { labelListLimit } from '@/lib/constants'
import MiniSearch from 'minisearch'

const lastGraph: any = {
  graph: null,
  searchEngine: null,
  labels: []
}

const GraphLabels = () => {
  const label = useSettingsStore.use.queryLabel()
  const graph = useGraphStore.use.sigmaGraph()

  const getSearchEngine = useCallback(async () => {
    if (lastGraph.graph == graph) {
      return {
        labels: lastGraph.labels,
        searchEngine: lastGraph.searchEngine
      }
    }
    const labels = ['*'].concat(await getGraphLabels())

    // Ensure query label exists
    if (!labels.includes(useSettingsStore.getState().queryLabel)) {
      useSettingsStore.getState().setQueryLabel(labels[0])
    }

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
    const documents = labels.map((str, index) => ({ id: index, value: str }))
    searchEngine.addAll(documents)

    lastGraph.graph = graph
    lastGraph.searchEngine = searchEngine
    lastGraph.labels = labels

    return {
      labels,
      searchEngine
    }
  }, [graph])

  const fetchData = useCallback(
    async (query?: string): Promise<string[]> => {
      const { labels, searchEngine } = await getSearchEngine()

      let result: string[] = labels
      if (query) {
        // Search labels
        result = searchEngine.search(query).map((r) => labels[r.id])
      }

      return result.length <= labelListLimit
        ? result
        : [...result.slice(0, labelListLimit), `And ${result.length - labelListLimit} others`]
    },
    [getSearchEngine]
  )

  const setQueryLabel = useCallback((label: string) => {
    if (label.startsWith('And ') && label.endsWith(' others')) return
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
