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
