import type { UndirectedGraph } from 'graphology'

// Helpers that work around two graphology 0.26.0 bugs, both rooted in the same
// thing: a node's adjacency is a PLAIN object keyed by neighbor names, and our
// node ids are extracted entity names that can collide with JS internals
// ('constructor', 'toString', '__proto__', 'hasOwnProperty', 'source', ...).

// Prototype-safe replacement for `graph.hasEdge(source, target)`.
//
// Bug #1 (why not hasEdge): the two-arg hasEdge does an unguarded
// `nodeData.undirected.hasOwnProperty(target)` METHOD call, so once a node gains
// a neighbor literally named 'hasOwnProperty' that adjacency entry shadows the
// method and every subsequent hasEdge on that node throws
// "hasOwnProperty is not a function" — aborting the whole graph build.
//
// `graph.edge(source, target)` reaches the same adjacency by property lookup
// (`sourceData.undirected[target]`) instead of a method call, so it never throws
// and returns the edge key (or undefined). But property lookup walks the
// prototype chain, which opens bug #2: adding an edge to a neighbor literally
// named '__proto__' runs `adjacency['__proto__'] = edgeData`, reassigning that
// adjacency object's PROTOTYPE to the EdgeData instance. Afterwards
// `edge(source, 'source')` / `edge(source, 'target')` read the inherited
// EdgeData.source/.target node objects and return a real-looking node key for an
// edge that does NOT exist — a phantom duplicate that makes callers silently
// drop a genuine edge to an entity named 'source'/'target'.
//
// So we do not trust edge()'s key blindly: the Map-backed single-arg
// `hasEdge(edgeKey)` confirms the returned key is a genuine edge (a Map lookup
// walks no prototype chain), collapsing any phantom node key to false. The
// hasNode guards keep this a true drop-in for hasEdge (edge() throws NotFound on
// an absent endpoint, whereas hasEdge returns false); hasNode is Map-backed and
// prototype-safe.
export const hasEdgeSafe = (graph: UndirectedGraph, source: string, target: string): boolean => {
  if (!graph.hasNode(source) || !graph.hasNode(target)) return false
  const edgeKey = graph.edge(source, target)
  return edgeKey !== undefined && graph.hasEdge(edgeKey)
}

// Add an undirected edge, working around bug #3 in the same family: the non-multi
// duplicate check does a bare `adjacency[target]` object lookup, so a TARGET
// node named like an Object.prototype property ('constructor', 'toString',
// '__proto__', 'hasOwnProperty', ...) makes addEdge throw "already exists" even
// though no such edge exists. The broken check only runs on the source side, and
// on an undirected graph the flipped edge is the same edge — so adding it
// reversed recovers it. Returns the dynamic edge id, or null if both orientations
// fail (e.g. both endpoints are prototype-named, or the edge is a genuine
// duplicate). Callers are expected to have pre-checked hasEdgeSafe, so a real
// duplicate is never masked.
export const addUndirectedEdgeSafe = (
  graph: UndirectedGraph,
  source: string,
  target: string,
  attributes: Record<string, unknown>
): string | null => {
  try {
    return graph.addEdge(source, target, attributes)
  } catch {
    try {
      return graph.addEdge(target, source, attributes)
    } catch {
      return null
    }
  }
}
