import { useRegisterEvents, useSetSettings, useSigma } from '@react-sigma/core'
import { useLayoutForceAtlas2 } from '@react-sigma/layout-forceatlas2'
import type { AbstractGraph } from 'graphology-types'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { Sigma } from 'sigma'

import type { EdgeType, NodeType } from '@/hooks/useLightragGraph'
import useTheme from '@/hooks/useTheme'
import * as Constants from '@/lib/constants'
import { useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'

const isButtonPressed = (ev: MouseEvent | TouchEvent) => {
  if (ev.type.startsWith('mouse')) {
    return (ev as MouseEvent).buttons !== 0
  }
  return false
}

const GraphControl = ({ disableHoverEffect }: { disableHoverEffect?: boolean }) => {
  const sigma = useSigma<NodeType, EdgeType>()
  const registerEvents = useRegisterEvents<NodeType, EdgeType>()
  const setSettings = useSetSettings<NodeType, EdgeType>()

  const maxIterations = useSettingsStore.use.graphLayoutMaxIterations()
  const { assign: assignLayout } = useLayoutForceAtlas2({
    iterations: maxIterations,
  })

  const { theme } = useTheme()
  const hideUnselectedEdges = useSettingsStore.use.enableHideUnselectedEdges()
  const enableEdgeEvents = useSettingsStore.use.enableEdgeEvents()
  const renderEdgeLabels = useSettingsStore.use.showEdgeLabel()
  const renderLabels = useSettingsStore.use.showNodeLabel()
  const minEdgeSize = useSettingsStore.use.minEdgeSize()
  const maxEdgeSize = useSettingsStore.use.maxEdgeSize()
  const selectedNode = useGraphStore.use.selectedNode()
  const focusedNode = useGraphStore.use.focusedNode()
  const selectedEdge = useGraphStore.use.selectedEdge()
  const focusedEdge = useGraphStore.use.focusedEdge()
  const sigmaGraph = useGraphStore.use.sigmaGraph()

  // Track system theme changes when theme is set to 'system'
  const [_systemThemeIsDark, setSystemThemeIsDark] = useState(
    () => window.matchMedia('(prefers-color-scheme: dark)').matches
  )

  useEffect(() => {
    if (theme === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
      const handler = (e: MediaQueryListEvent) => setSystemThemeIsDark(e.matches)
      mediaQuery.addEventListener('change', handler)
      return () => mediaQuery.removeEventListener('change', handler)
    }
  }, [theme])

  // ==================== PERFORMANCE OPTIMIZATION ====================
  // Cache selection state in refs to avoid recreating reducers on every hover
  const selectionRef = useRef({
    selectedNode: null as string | null,
    focusedNode: null as string | null,
    selectedEdge: null as string | null,
    focusedEdge: null as string | null,
    hideUnselectedEdges,
  })

  // Cache computed neighbors for the focused node (Set for O(1) lookup)
  const neighborsCache = useRef<{ nodeId: string | null; neighbors: Set<string> }>({
    nodeId: null,
    neighbors: new Set(),
  })

  // Memoize theme-derived values to avoid recomputing in reducers
  const themeColors = useMemo(() => {
    const isDarkTheme =
      theme === 'dark' ||
      (theme === 'system' && window.document.documentElement.classList.contains('dark'))
    return {
      isDarkTheme,
      labelColor: isDarkTheme ? Constants.labelColorDarkTheme : undefined,
      edgeColor: isDarkTheme ? Constants.edgeColorDarkTheme : undefined,
      edgeHighlightColor: isDarkTheme
        ? Constants.edgeColorHighlightedDarkTheme
        : Constants.edgeColorHighlightedLightTheme,
    }
  }, [theme])

  // Update refs when selection changes, then trigger sigma refresh (not reducer recreation)
  useEffect(() => {
    selectionRef.current = {
      selectedNode,
      focusedNode,
      selectedEdge,
      focusedEdge,
      hideUnselectedEdges,
    }

    // Invalidate and rebuild neighbor cache if focused node changed
    const targetNode = focusedNode || selectedNode
    if (neighborsCache.current.nodeId !== targetNode) {
      if (targetNode && sigma) {
        const graph = sigma.getGraph()
        if (graph.hasNode(targetNode)) {
          // Build Set once for O(1) lookups in reducer
          neighborsCache.current = {
            nodeId: targetNode,
            neighbors: new Set(graph.neighbors(targetNode)),
          }
        } else {
          neighborsCache.current = { nodeId: null, neighbors: new Set() }
        }
      } else {
        neighborsCache.current = { nodeId: null, neighbors: new Set() }
      }
    }

    // Trigger sigma refresh to re-run reducers with updated ref values
    if (sigma) {
      sigma.refresh()
    }
  }, [selectedNode, focusedNode, selectedEdge, focusedEdge, hideUnselectedEdges, sigma])
  // ==================== END OPTIMIZATION ====================

  /**
   * When component mount or maxIterations changes
   * => ensure graph reference and apply layout
   */
  useEffect(() => {
    if (sigmaGraph && sigma) {
      try {
        if (typeof sigma.setGraph === 'function') {
          sigma.setGraph(sigmaGraph as unknown as AbstractGraph<NodeType, EdgeType>)
          console.log('Binding graph to sigma instance')
        } else {
          // Type assertion for backward compatibility with older sigma versions
          ;(sigma as unknown as { graph: typeof sigmaGraph }).graph = sigmaGraph
          console.warn('Sigma missing setGraph function, set graph property directly')
        }
      } catch (error) {
        console.error('Error setting graph on sigma instance:', error)
      }

      assignLayout()
      console.log('Initial layout applied to graph')
    }
  }, [sigma, sigmaGraph, assignLayout])

  /**
   * Ensure the sigma instance is set in the store
   */
  useEffect(() => {
    if (sigma) {
      const currentInstance = useGraphStore.getState().sigmaInstance
      if (!currentInstance) {
        console.log('Setting sigma instance from GraphControl')
        // Cast to generic Sigma type for store compatibility
        // The specific NodeType/EdgeType typing is preserved in the component context
        useGraphStore.getState().setSigmaInstance(sigma as unknown as Sigma)
      }
    }
  }, [sigma])

  /**
   * Register events
   */
  useEffect(() => {
    const { setFocusedNode, setSelectedNode, setFocusedEdge, setSelectedEdge, clearSelection } =
      useGraphStore.getState()

    interface NodeEvent {
      node: string
      event: { original: MouseEvent | TouchEvent }
    }
    interface EdgeEvent {
      edge: string
      event: { original: MouseEvent | TouchEvent }
    }
    type EventHandler = ((e: NodeEvent) => void) | ((e: EdgeEvent) => void) | (() => void)

    const events: Record<string, EventHandler> = {
      enterNode: (event: NodeEvent) => {
        if (!isButtonPressed(event.event.original)) {
          const graph = sigma.getGraph()
          if (graph.hasNode(event.node)) {
            setFocusedNode(event.node)
          }
        }
      },
      leaveNode: (event: NodeEvent) => {
        if (!isButtonPressed(event.event.original)) {
          setFocusedNode(null)
        }
      },
      clickNode: (event: NodeEvent) => {
        const graph = sigma.getGraph()
        if (graph.hasNode(event.node)) {
          setSelectedNode(event.node)
          setSelectedEdge(null)
        }
      },
      clickStage: () => clearSelection(),
    }

    if (enableEdgeEvents) {
      events.clickEdge = (event: EdgeEvent) => {
        setSelectedEdge(event.edge)
        setSelectedNode(null)
      }
      events.enterEdge = (event: EdgeEvent) => {
        if (!isButtonPressed(event.event.original)) {
          setFocusedEdge(event.edge)
        }
      }
      events.leaveEdge = (event: EdgeEvent) => {
        if (!isButtonPressed(event.event.original)) {
          setFocusedEdge(null)
        }
      }
    }

    registerEvents(events as Parameters<typeof registerEvents>[0])

    return () => {
      try {
        console.log('Cleaning up graph event listeners')
      } catch (error) {
        console.warn('Error cleaning up graph event listeners:', error)
      }
    }
  }, [registerEvents, enableEdgeEvents, sigma])

  /**
   * Recalculate edge sizes when settings change
   */
  useEffect(() => {
    if (sigma && sigmaGraph) {
      const graph = sigma.getGraph()

      let minWeight = Number.MAX_SAFE_INTEGER
      let maxWeight = 0

      graph.forEachEdge((edge) => {
        const weight = graph.getEdgeAttribute(edge, 'originalWeight') || 1
        if (typeof weight === 'number') {
          minWeight = Math.min(minWeight, weight)
          maxWeight = Math.max(maxWeight, weight)
        }
      })

      const weightRange = maxWeight - minWeight
      if (weightRange > 0) {
        const sizeScale = maxEdgeSize - minEdgeSize
        graph.forEachEdge((edge) => {
          const weight = graph.getEdgeAttribute(edge, 'originalWeight') || 1
          if (typeof weight === 'number') {
            const scaledSize = minEdgeSize + sizeScale * ((weight - minWeight) / weightRange) ** 0.5
            graph.setEdgeAttribute(edge, 'size', scaledSize)
          }
        })
      } else {
        graph.forEachEdge((edge) => {
          graph.setEdgeAttribute(edge, 'size', minEdgeSize)
        })
      }

      sigma.refresh()
    }
  }, [sigma, sigmaGraph, minEdgeSize, maxEdgeSize])

  // ==================== STABLE REDUCERS (read from refs) ====================
  // These reducers are stable and only recreated when sigma/theme changes
  // Selection state is read from refs, avoiding costly reducer recreation on hover

  const nodeReducer = useCallback(
    (node: string, data: NodeType) => {
      const graph = sigma.getGraph()
      const { labelColor, isDarkTheme } = themeColors
      const { selectedNode, focusedNode, selectedEdge, focusedEdge } = selectionRef.current

      if (!graph.hasNode(node)) {
        return { ...data, highlighted: false, labelColor }
      }

      // Always start with highlighted: false to prevent persistence
      const newData: NodeType & {
        labelColor?: string
        borderColor?: string
        borderSize?: number
      } = { ...data, highlighted: false, labelColor }

      // Hidden connections indicator
      const dbDegree = graph.getNodeAttribute(node, 'db_degree') || 0
      const visualDegree = graph.degree(node)
      if (dbDegree > visualDegree) {
        newData.borderColor = Constants.nodeBorderColorHiddenConnections
        newData.borderSize = 1.5
      }

      if (disableHoverEffect) {
        return newData
      }

      const _focusedNode = focusedNode || selectedNode
      const _focusedEdge = focusedEdge || selectedEdge

      if (_focusedNode && graph.hasNode(_focusedNode)) {
        // O(1) lookup using cached Set instead of O(n) array.includes()
        const isNeighbor = node === _focusedNode || neighborsCache.current.neighbors.has(node)

        if (isNeighbor) {
          newData.highlighted = true
          if (node === selectedNode) {
            newData.borderColor = Constants.nodeBorderColorSelected
          }
        }
      } else if (_focusedEdge && graph.hasEdge(_focusedEdge)) {
        if (graph.extremities(_focusedEdge).includes(node)) {
          newData.highlighted = true
          newData.size = 3
        }
      } else {
        // No focus - early return with original colors (don't apply disabled)
        return newData
      }

      // Apply highlight/disabled styling only when there's a focus target
      if (newData.highlighted) {
        if (isDarkTheme) {
          newData.labelColor = Constants.LabelColorHighlightedDarkTheme
        }
      } else {
        newData.color = Constants.nodeColorDisabled
      }

      return newData
    },
    [sigma, disableHoverEffect, themeColors]
  )

  const edgeReducer = useCallback(
    (edge: string, data: EdgeType) => {
      const graph = sigma.getGraph()
      const { labelColor, edgeColor, edgeHighlightColor } = themeColors
      const { selectedNode, focusedNode, selectedEdge, focusedEdge, hideUnselectedEdges } =
        selectionRef.current

      if (!graph.hasEdge(edge)) {
        return { ...data, hidden: false, labelColor, color: edgeColor }
      }

      const newData = { ...data, hidden: false, labelColor, color: edgeColor }

      if (disableHoverEffect) {
        return newData
      }

      const _focusedNode = focusedNode || selectedNode

      if (_focusedNode && graph.hasNode(_focusedNode)) {
        const includesNode = graph.extremities(edge).includes(_focusedNode)
        if (hideUnselectedEdges && !includesNode) {
          newData.hidden = true
        } else if (includesNode) {
          newData.color = edgeHighlightColor
        }
      } else {
        const _selectedEdge = selectedEdge && graph.hasEdge(selectedEdge) ? selectedEdge : null
        const _focusedEdge = focusedEdge && graph.hasEdge(focusedEdge) ? focusedEdge : null

        if (_selectedEdge || _focusedEdge) {
          if (edge === _selectedEdge) {
            newData.color = Constants.edgeColorSelected
          } else if (edge === _focusedEdge) {
            newData.color = edgeHighlightColor
          } else if (hideUnselectedEdges) {
            newData.hidden = true
          }
        }
      }

      return newData
    },
    [sigma, disableHoverEffect, themeColors]
  )

  // Set reducers only when they actually change (not on every hover)
  useEffect(() => {
    setSettings({
      enableEdgeEvents,
      renderEdgeLabels,
      renderLabels,
      nodeReducer,
      edgeReducer,
    })
  }, [setSettings, enableEdgeEvents, renderEdgeLabels, renderLabels, nodeReducer, edgeReducer])

  return null
}

export default GraphControl
