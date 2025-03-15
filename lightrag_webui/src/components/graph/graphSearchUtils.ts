import { DirectedGraph } from 'graphology'
import MiniSearch from 'minisearch'

export const messageId = '__message_item'

// Reset this cache when graph changes to ensure fresh search results
export const searchCache: {
  graph: DirectedGraph | null;
  searchEngine: MiniSearch | null;
} = {
  graph: null,
  searchEngine: null
}

export const updateSearchEngine = (nodeId: string, graph: DirectedGraph) => {
  if (!searchCache.searchEngine || !graph) return

  const newDocument = {
    id: nodeId,
    label: graph.getNodeAttribute(nodeId, 'label')
  }
  searchCache.searchEngine.add(newDocument)
}

export const removeFromSearchEngine = (nodeId: string) => {
  if (!searchCache.searchEngine) return
  searchCache.searchEngine.remove({ id: nodeId })
}
