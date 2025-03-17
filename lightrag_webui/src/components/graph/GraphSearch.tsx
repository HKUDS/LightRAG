import { FC, useCallback, useEffect, useMemo } from 'react'
import {
  EdgeById,
  NodeById,
  GraphSearchInputProps,
  GraphSearchContextProviderProps
} from '@react-sigma/graph-search'
import { AsyncSearch } from '@/components/ui/AsyncSearch'
import { searchResultLimit } from '@/lib/constants'
import { useGraphStore } from '@/stores/graph'
import MiniSearch from 'minisearch'
import { useTranslation } from 'react-i18next'

interface OptionItem {
  id: string
  type: 'nodes' | 'edges' | 'message'
  message?: string
}

function OptionComponent(item: OptionItem) {
  return (
    <div>
      {item.type === 'nodes' && <NodeById id={item.id} />}
      {item.type === 'edges' && <EdgeById id={item.id} />}
      {item.type === 'message' && <div>{item.message}</div>}
    </div>
  )
}

const messageId = '__message_item'
// Reset this cache when graph changes to ensure fresh search results
const lastGraph: any = {
  graph: null,
  searchEngine: null
}

/**
 * Component thats display the search input.
 */
export const GraphSearchInput = ({
  onChange,
  onFocus,
  value
}: {
  onChange: GraphSearchInputProps['onChange']
  onFocus?: GraphSearchInputProps['onFocus']
  value?: GraphSearchInputProps['value']
}) => {
  const { t } = useTranslation()
  const graph = useGraphStore.use.sigmaGraph()

  // Force reset the cache when graph changes
  useEffect(() => {
    if (graph) {
      // Reset cache to ensure fresh search results with new graph data
      lastGraph.graph = null;
      lastGraph.searchEngine = null;
    }
  }, [graph]);

  const searchEngine = useMemo(() => {
    if (lastGraph.graph == graph) {
      return lastGraph.searchEngine
    }
    if (!graph || graph.nodes().length == 0) return

    lastGraph.graph = graph

    const searchEngine = new MiniSearch({
      idField: 'id',
      fields: ['label'],
      searchOptions: {
        prefix: true,
        fuzzy: 0.2,
        boost: {
          label: 2
        }
      }
    })

    // Add documents
    const documents = graph.nodes().map((id: string) => ({
      id: id,
      label: graph.getNodeAttribute(id, 'label')
    }))
    searchEngine.addAll(documents)

    lastGraph.searchEngine = searchEngine
    return searchEngine
  }, [graph])

  /**
   * Loading the options while the user is typing.
   */
  const loadOptions = useCallback(
    async (query?: string): Promise<OptionItem[]> => {
      if (onFocus) onFocus(null)
      if (!graph || !searchEngine) return []

      // If no query, return first searchResultLimit nodes
      if (!query) {
        const nodeIds = graph.nodes().slice(0, searchResultLimit)
        return nodeIds.map(id => ({
          id,
          type: 'nodes'
        }))
      }

      // If has query, search nodes
      const result: OptionItem[] = searchEngine.search(query).map((r: { id: string }) => ({
        id: r.id,
        type: 'nodes'
      }))

      // prettier-ignore
      return result.length <= searchResultLimit
        ? result
        : [
          ...result.slice(0, searchResultLimit),
          {
            type: 'message',
            id: messageId,
            message: t('graphPanel.search.message', { count: result.length - searchResultLimit })
          }
        ]
    },
    [graph, searchEngine, onFocus, t]
  )

  return (
    <AsyncSearch
      className="bg-background/60 w-24 rounded-xl border-1 opacity-60 backdrop-blur-lg transition-all hover:w-fit hover:opacity-100"
      fetcher={loadOptions}
      renderOption={OptionComponent}
      getOptionValue={(item) => item.id}
      value={value && value.type !== 'message' ? value.id : null}
      onChange={(id) => {
        if (id !== messageId) onChange(id ? { id, type: 'nodes' } : null)
      }}
      onFocus={(id) => {
        if (id !== messageId && onFocus) onFocus(id ? { id, type: 'nodes' } : null)
      }}
      label={'item'}
      placeholder={t('graphPanel.search.placeholder')}
    />
  )
}

/**
 * Component that display the search.
 */
const GraphSearch: FC<GraphSearchInputProps & GraphSearchContextProviderProps> = ({ ...props }) => {
  return <GraphSearchInput {...props} />
}

export default GraphSearch
