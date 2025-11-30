import { AsyncSearch } from '@/components/ui/AsyncSearch'
import { searchResultLimit } from '@/lib/constants'
import { useGraphStore } from '@/stores/graph'
import {
  EdgeById,
  type GraphSearchContextProviderProps,
  type GraphSearchInputProps,
} from '@react-sigma/graph-search'
import MiniSearch from 'minisearch'
import { type FC, useCallback, useEffect } from 'react'
import { useTranslation } from 'react-i18next'

// Message item identifier for search results
export const messageId = '__message_item'

// Search result option item interface
export interface OptionItem {
  id: string
  type: 'nodes' | 'edges' | 'message'
  message?: string
}

const NodeOption = ({ id }: { id: string }) => {
  const graph = useGraphStore.use.sigmaGraph()

  // Early return if no graph or node doesn't exist
  if (!graph?.hasNode(id)) {
    return null
  }

  // Safely get node attributes with fallbacks
  const label = graph.getNodeAttribute(id, 'label') || id
  const color = graph.getNodeAttribute(id, 'color') || '#666'
  const size = graph.getNodeAttribute(id, 'size') || 4

  // Custom node display component that doesn't rely on @react-sigma/graph-search
  return (
    <div className="flex items-center gap-2 p-2 text-sm">
      <div
        className="rounded-full flex-shrink-0"
        style={{
          width: Math.max(8, Math.min(size * 2, 16)),
          height: Math.max(8, Math.min(size * 2, 16)),
          backgroundColor: color,
        }}
      />
      <span className="truncate">{label}</span>
    </div>
  )
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
  value,
}: {
  onChange: GraphSearchInputProps['onChange']
  onFocus?: GraphSearchInputProps['onFocus']
  value?: GraphSearchInputProps['value']
}) => {
  const { t } = useTranslation()
  const graph = useGraphStore.use.sigmaGraph()
  const searchEngine = useGraphStore.use.searchEngine()

  // Reset search engine when graph changes
  useEffect(() => {
    if (graph) {
      useGraphStore.getState().resetSearchEngine()
    }
  }, [graph])

  // Create search engine when needed
  useEffect(() => {
    // Skip if no graph, empty graph, or search engine already exists
    if (!graph || graph.nodes().length === 0 || searchEngine) {
      return
    }

    // Create new search engine
    const newSearchEngine = new MiniSearch({
      idField: 'id',
      fields: ['label'],
      searchOptions: {
        prefix: true,
        fuzzy: 0.2,
        boost: {
          label: 2,
        },
      },
    })

    // Add nodes to search engine with safety checks
    const documents = graph
      .nodes()
      .filter((id) => graph.hasNode(id)) // Ensure node exists before accessing attributes
      .map((id: string) => ({
        id: id,
        label: graph.getNodeAttribute(id, 'label'),
      }))

    if (documents.length > 0) {
      newSearchEngine.addAll(documents)
    }

    // Update search engine in store
    useGraphStore.getState().setSearchEngine(newSearchEngine)
  }, [graph, searchEngine])

  /**
   * Loading the options while the user is typing.
   */
  const loadOptions = useCallback(
    async (query?: string): Promise<OptionItem[]> => {
      if (onFocus) onFocus(null)

      // Safety checks to prevent crashes
      if (!graph || !searchEngine) {
        return []
      }

      // Verify graph has nodes before proceeding
      if (graph.nodes().length === 0) {
        return []
      }

      // If no query, return some nodes for user to select
      if (!query) {
        const nodeIds = graph
          .nodes()
          .filter((id) => graph.hasNode(id))
          .slice(0, searchResultLimit)
        return nodeIds.map((id) => ({
          id,
          type: 'nodes',
        }))
      }

      // If has query, search nodes and verify they still exist
      let result: OptionItem[] = searchEngine
        .search(query)
        .filter((r: { id: string }) => graph.hasNode(r.id))
        .map((r: { id: string }) => ({
          id: r.id,
          type: 'nodes',
        }))

      // Add middle-content matching if results are few
      // This enables matching content in the middle of text, not just from the beginning
      if (result.length < 5) {
        // Get already matched IDs to avoid duplicates
        const matchedIds = new Set(result.map((item) => item.id))

        // Perform middle-content matching on all nodes with safety checks
        const middleMatchResults = graph
          .nodes()
          .filter((id) => {
            // Skip already matched nodes
            if (matchedIds.has(id)) return false

            // Ensure node exists before accessing attributes
            if (!graph.hasNode(id)) return false

            // Get node label safely
            const label = graph.getNodeAttribute(id, 'label')
            // Match if label contains query string but doesn't start with it
            return (
              label &&
              typeof label === 'string' &&
              !label.toLowerCase().startsWith(query.toLowerCase()) &&
              label.toLowerCase().includes(query.toLowerCase())
            )
          })
          .map((id) => ({
            id,
            type: 'nodes' as const,
          }))

        // Merge results
        result = [...result, ...middleMatchResults]
      }

      // prettier-ignore
      return result.length <= searchResultLimit
        ? result
        : [
            ...result.slice(0, searchResultLimit),
            {
              type: 'message',
              id: messageId,
              message: t('graphPanel.search.message', { count: result.length - searchResultLimit }),
            },
          ]
    },
    [graph, searchEngine, onFocus, t]
  )

  return (
    <AsyncSearch
      className="bg-background/60 w-24 rounded-xl border-1 opacity-60 backdrop-blur-lg transition-all hover:w-fit hover:opacity-100 w-full"
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
      ariaLabel={t('graphPanel.search.placeholder')}
      placeholder={t('graphPanel.search.placeholder')}
      noResultsMessage={t('graphPanel.search.placeholder')}
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
