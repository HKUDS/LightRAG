import { describe, expect, test } from 'bun:test'
import { UndirectedGraph } from 'graphology'

import { hasEdgeSafe, addUndirectedEdgeSafe } from '@/utils/graphEdgeSafety'

// Node ids come from extracted entity names, so they can collide with JS
// internals. These tests pin the graphology 0.26.0 prototype-name workarounds.
const graphWith = (...nodes: string[]): UndirectedGraph => {
  const g = new UndirectedGraph()
  for (const n of nodes) g.addNode(n)
  return g
}

// Mirrors the createSigmaGraph / expansion edge loop: cheap pre-check, then the
// addEdge backstop. Returns whether the edge ended up in the graph.
const buildEdge = (g: UndirectedGraph, source: string, target: string): 'added' | 'skipped' => {
  if (!g.hasNode(source) || !g.hasNode(target) || hasEdgeSafe(g, source, target)) return 'skipped'
  return addUndirectedEdgeSafe(g, source, target, {}) === null ? 'skipped' : 'added'
}

describe('hasEdgeSafe', () => {
  test('reports real / absent edges like hasEdge', () => {
    const g = graphWith('A', 'B', 'C')
    g.addEdge('A', 'B', {})
    expect(hasEdgeSafe(g, 'A', 'B')).toBe(true)
    expect(hasEdgeSafe(g, 'B', 'A')).toBe(true) // undirected: same edge
    expect(hasEdgeSafe(g, 'A', 'C')).toBe(false)
  })

  test('returns false (not throw) for absent endpoints', () => {
    const g = graphWith('A')
    expect(hasEdgeSafe(g, 'A', 'missing')).toBe(false)
    expect(hasEdgeSafe(g, 'missing', 'A')).toBe(false)
  })

  // Bug #1: original two-arg hasEdge throws once a node has a 'hasOwnProperty'
  // neighbor. hasEdgeSafe must not throw. (This is what the PR set out to fix.)
  test('does not throw when a node has a "hasOwnProperty" neighbor', () => {
    const g = graphWith('A', 'hasOwnProperty', 'B')
    addUndirectedEdgeSafe(g, 'A', 'hasOwnProperty', {})
    expect(() => g.hasEdge('A', 'B')).toThrow() // graphology's own bug, for the record
    expect(() => hasEdgeSafe(g, 'A', 'B')).not.toThrow()
    expect(hasEdgeSafe(g, 'A', 'B')).toBe(false)
  })

  // Bug #2 (regression the naive `edge() !== undefined` version had): a
  // '__proto__' neighbor pollutes the adjacency prototype, so edge(A,'source')
  // returns a phantom key. hasEdgeSafe must still report false.
  test('does not report phantom edges after a "__proto__" neighbor', () => {
    const g = graphWith('A', '__proto__', 'source', 'target')
    addUndirectedEdgeSafe(g, 'A', '__proto__', {})
    expect(hasEdgeSafe(g, 'A', 'source')).toBe(false)
    expect(hasEdgeSafe(g, 'A', 'target')).toBe(false)
    // the real edge to the prototype-named neighbor is still reported
    expect(hasEdgeSafe(g, 'A', '__proto__')).toBe(true)
  })
})

describe('addUndirectedEdgeSafe', () => {
  test('adds a normal edge and returns its key', () => {
    const g = graphWith('A', 'B')
    expect(addUndirectedEdgeSafe(g, 'A', 'B', {})).not.toBeNull()
    expect(g.size).toBe(1)
  })

  test('dedups a genuine duplicate (returns null, no double-add)', () => {
    const g = graphWith('A', 'B')
    expect(addUndirectedEdgeSafe(g, 'A', 'B', {})).not.toBeNull()
    expect(addUndirectedEdgeSafe(g, 'A', 'B', {})).toBeNull()
    expect(g.size).toBe(1)
  })

  test('recovers a prototype-named target via the flipped orientation', () => {
    const g = graphWith('A', '__proto__')
    expect(addUndirectedEdgeSafe(g, 'A', '__proto__', {})).not.toBeNull()
    expect(g.size).toBe(1)
  })
})

describe('build loop (pre-check + backstop) is prototype-safe end to end', () => {
  // The exact failure the codex review caught: a '__proto__' edge processed
  // before a real A->source edge must NOT cause the real edge to be dropped.
  test('keeps a real edge to "source"/"target" alongside a "__proto__" edge', () => {
    const g = graphWith('A', '__proto__', 'source', 'target')
    expect(buildEdge(g, 'A', '__proto__')).toBe('added')
    expect(buildEdge(g, 'A', 'source')).toBe('added')
    expect(buildEdge(g, 'A', 'target')).toBe('added')
    expect(g.size).toBe(3)
  })

  test('still skips true duplicates', () => {
    const g = graphWith('A', 'B')
    expect(buildEdge(g, 'A', 'B')).toBe('added')
    expect(buildEdge(g, 'A', 'B')).toBe('skipped')
    expect(buildEdge(g, 'B', 'A')).toBe('skipped') // undirected
    expect(g.size).toBe(1)
  })

  test('does not abort on a "hasOwnProperty" neighbor', () => {
    const g = graphWith('A', 'hasOwnProperty', 'B')
    expect(buildEdge(g, 'A', 'hasOwnProperty')).toBe('added')
    let result: 'added' | 'skipped' | undefined
    expect(() => {
      result = buildEdge(g, 'A', 'B')
    }).not.toThrow()
    expect(result).toBe('added')
    expect(g.size).toBe(2)
  })
})
