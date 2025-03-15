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
import { OptionItem } from './graphSearchTypes'
import { messageId, searchCache } from './graphSearchUtils'

const NodeOption = ({ id }: { id: string }) => {
  const graph = useGraphStore.use.sigmaGraph()
  if (!graph?.hasNode(id)) {
    return null
  }
  return <NodeById id={id} />
}

function OptionComponent(item: OptionItem) {
  return (
    <div>
      {item.type === 'nodes' && <NodeOption id={item.id} />}
      {item.type === 'edges' && <EdgeById id={item.id} />}
      {item.type === 'message' && <div>{item.message}</div>}
    </div>
  )
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
      searchCache.graph = null;
      searchCache.searchEngine = null;
    }
  }, [graph]);

  const searchEngine = useMemo(() => {
    if (searchCache.graph == graph) {
      return searchCache.searchEngine
    }
    if (!graph || graph.nodes().length == 0) return

    searchCache.graph = graph

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

    searchCache.searchEngine = searchEngine
    return searchEngine
  }, [graph])

  /**
   * Loading the options while the user is typing.
   */
  const loadOptions = useCallback(
    async (query?: string): Promise<OptionItem[]> => {
      if (onFocus) onFocus(null)
      
      // Safety checks to prevent crashes
      if (!graph || !searchEngine) {
        // Reset cache to ensure fresh search engine initialization on next render
        searchCache.graph = null
        searchCache.searchEngine = null
        return []
      }

      // Verify graph has nodes before proceeding
      if (graph.nodes().length === 0) {
        return []
      }

      // If no query, return first searchResultLimit nodes that exist
      if (!query) {
        const nodeIds = graph.nodes()
          .filter(id => graph.hasNode(id))
          .slice(0, searchResultLimit)
        return nodeIds.map(id => ({
          id,
          type: 'nodes'
        }))
      }

      // If has query, search nodes and verify they still exist
      const result: OptionItem[] = searchEngine.search(query)
        .filter((r: { id: string }) => graph.hasNode(r.id))
        .map((r: { id: string }) => ({
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
