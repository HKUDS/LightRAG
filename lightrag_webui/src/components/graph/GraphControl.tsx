import { useRegisterEvents, useSetSettings, useSigma } from '@react-sigma/core'
// import { useLayoutCircular } from '@react-sigma/layout-circular'
import { useLayoutForceAtlas2 } from '@react-sigma/layout-forceatlas2'
import type { AbstractGraph } from 'graphology-types'
import { useCallback, useEffect, useRef, useState } from 'react'

// import useRandomGraph, { EdgeType, NodeType } from '@/hooks/useRandomGraph'
import type { EdgeType, NodeType } from '@/hooks/useLightragGraph'
import useTheme from '@/hooks/useTheme'
import * as Constants from '@/lib/constants'

import { useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'

const isButtonPressed = (ev: MouseEvent | TouchEvent) => {
  if (ev.type.startsWith('mouse')) {
    if ((ev as MouseEvent).buttons !== 0) {
      return true
    }
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
  const [systemThemeIsDark, setSystemThemeIsDark] = useState(
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

  /**
   * When component mount or maxIterations changes
   * => ensure graph reference and apply layout
   */
  useEffect(() => {
    if (sigmaGraph && sigma) {
      // Ensure sigma binding to sigmaGraph
      try {
        if (typeof sigma.setGraph === 'function') {
          sigma.setGraph(sigmaGraph as unknown as AbstractGraph<NodeType, EdgeType>)
          console.log('Binding graph to sigma instance')
        } else {
          ;(sigma as any).graph = sigmaGraph
          console.warn('Simgma missing setGraph function, set graph property directly')
        }
      } catch (error) {
        console.error('Error setting graph on sigma instance:', error)
      }

      assignLayout()
      console.log('Initial layout applied to graph')
    }
  }, [sigma, sigmaGraph, assignLayout, maxIterations])

  /**
   * Ensure the sigma instance is set in the store
   * This provides a backup in case the instance wasn't set in GraphViewer
   */
  useEffect(() => {
    if (sigma) {
      // Double-check that the store has the sigma instance
      const currentInstance = useGraphStore.getState().sigmaInstance
      if (!currentInstance) {
        console.log('Setting sigma instance from GraphControl')
        useGraphStore.getState().setSigmaInstance(sigma)
      }
    }
  }, [sigma])

  /**
   * When component mount
   * => register events
   */
  useEffect(() => {
    const { setFocusedNode, setSelectedNode, setFocusedEdge, setSelectedEdge, clearSelection } =
      useGraphStore.getState()

    // Define event types
    type NodeEvent = { node: string; event: { original: MouseEvent | TouchEvent } }
    type EdgeEvent = { edge: string; event: { original: MouseEvent | TouchEvent } }

    // Register all events, but edge events will only be processed if enableEdgeEvents is true
    const events: Record<string, any> = {
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

    // Only add edge event handlers if enableEdgeEvents is true
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

    // Register the events
    registerEvents(events)

    // Cleanup function - basic cleanup without relying on specific APIs
    return () => {
      try {
        console.log('Cleaning up graph event listeners')
      } catch (error) {
        console.warn('Error cleaning up graph event listeners:', error)
      }
    }
  }, [registerEvents, enableEdgeEvents, sigma])

  /**
   * When edge size settings change, recalculate edge sizes and refresh the sigma instance
   * to ensure changes take effect immediately
   */
  useEffect(() => {
    if (sigma && sigmaGraph) {
      // Get the graph from sigma
      const graph = sigma.getGraph()

      // Find min and max weight values
      let minWeight = Number.MAX_SAFE_INTEGER
      let maxWeight = 0

      graph.forEachEdge((edge) => {
        // Get original weight (before scaling)
        const weight = graph.getEdgeAttribute(edge, 'originalWeight') || 1
        if (typeof weight === 'number') {
          minWeight = Math.min(minWeight, weight)
          maxWeight = Math.max(maxWeight, weight)
        }
      })

      // Scale edge sizes based on weight range and current min/max edge size settings
      const weightRange = maxWeight - minWeight
      if (weightRange > 0) {
        const sizeScale = maxEdgeSize - minEdgeSize
        graph.forEachEdge((edge) => {
          const weight = graph.getEdgeAttribute(edge, 'originalWeight') || 1
          if (typeof weight === 'number') {
            const scaledSize =
              minEdgeSize + sizeScale * Math.pow((weight - minWeight) / weightRange, 0.5)
            graph.setEdgeAttribute(edge, 'size', scaledSize)
          }
        })
      } else {
        // If all weights are the same, use default size
        graph.forEachEdge((edge) => {
          graph.setEdgeAttribute(edge, 'size', minEdgeSize)
        })
      }

      // Refresh the sigma instance to apply changes
      sigma.refresh()
    }
  }, [sigma, sigmaGraph, minEdgeSize, maxEdgeSize])

  // Cache selection state in refs so we don't trigger expensive reducer recreation on hover/selection changes
  const selectionRef = useRef({
    selectedNode: null as string | null,
    focusedNode: null as string | null,
    selectedEdge: null as string | null,
    focusedEdge: null as string | null,
    hideUnselectedEdges,
  })

  const focusedNeighborsRef = useRef<{ key: string | null; neighbors: Set<string> | null }>({
    key: null,
    neighbors: null,
  })

  useEffect(() => {
    selectionRef.current = {
      selectedNode,
      focusedNode,
      selectedEdge,
      focusedEdge,
      hideUnselectedEdges,
    }

    // Invalidate neighbor cache when focused node changes
    if (focusedNeighborsRef.current.key !== (focusedNode || null)) {
      focusedNeighborsRef.current = { key: focusedNode || null, neighbors: null }
    }
  }, [selectedNode, focusedNode, selectedEdge, focusedEdge, hideUnselectedEdges])

  // Theme values used inside reducers; kept in refs to avoid re-creating reducer functions
  const themeRef = useRef({
    isDarkTheme: false,
    labelColor: undefined as string | undefined,
    edgeColor: undefined as string | undefined,
  })

  useEffect(() => {
    const isDarkTheme =
      theme === 'dark' ||
      (theme === 'system' && window.document.documentElement.classList.contains('dark'))

    themeRef.current = {
      isDarkTheme,
      labelColor: isDarkTheme ? Constants.labelColorDarkTheme : undefined,
      edgeColor: isDarkTheme ? Constants.edgeColorDarkTheme : undefined,
    }
  }, [theme, systemThemeIsDark])

  // Helper to lazily compute focused node neighbors and reuse across reducer calls
  const getFocusedNeighbors = useCallback((graph: AbstractGraph, nodeId: string): Set<string> => {
    if (focusedNeighborsRef.current.key === nodeId && focusedNeighborsRef.current.neighbors) {
      return focusedNeighborsRef.current.neighbors
    }
    const neighbors = new Set(graph.neighbors(nodeId))
    focusedNeighborsRef.current = { key: nodeId, neighbors }
    return neighbors
  }, [])

  const nodeReducer = useCallback(
    (node: string, data: NodeType) => {
      const graph = sigma.getGraph()
      const { selectedNode, focusedNode, selectedEdge, focusedEdge } = selectionRef.current
      const { isDarkTheme, labelColor } = themeRef.current

      if (!graph.hasNode(node)) {
        return { ...data, highlighted: false, labelColor }
      }

      const newData: NodeType & { labelColor?: string; borderColor?: string; borderSize?: number } =
        {
          ...data,
          highlighted: data.highlighted || false,
          labelColor,
        }

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

      const targetNode = focusedNode || selectedNode
      const targetEdge = focusedEdge || selectedEdge

      if (targetNode && graph.hasNode(targetNode)) {
        try {
          const neighbors = getFocusedNeighbors(graph, targetNode)
          if (node === targetNode || neighbors.has(node)) {
            newData.highlighted = true
            if (node === selectedNode) {
              newData.borderColor = Constants.nodeBorderColorSelected
            }
          }
        } catch (error) {
          console.error('Error in nodeReducer:', error)
          return { ...data, highlighted: false, labelColor }
        }
      } else if (targetEdge && graph.hasEdge(targetEdge)) {
        try {
          if (graph.extremities(targetEdge).includes(node)) {
            newData.highlighted = true
            newData.size = 3
          }
        } catch (error) {
          console.error('Error accessing edge extremities in nodeReducer:', error)
          return { ...data, highlighted: false, labelColor }
        }
      }

      if (newData.highlighted) {
        if (isDarkTheme) {
          newData.labelColor = Constants.LabelColorHighlightedDarkTheme
        }
      } else {
        newData.color = Constants.nodeColorDisabled
      }

      return newData
    },
    [sigma, disableHoverEffect, getFocusedNeighbors]
  )

  const edgeReducer = useCallback(
    (edge: string, data: EdgeType) => {
      const graph = sigma.getGraph()
      const { selectedNode, focusedNode, selectedEdge, focusedEdge, hideUnselectedEdges } =
        selectionRef.current
      const { isDarkTheme, labelColor, edgeColor } = themeRef.current

      if (!graph.hasEdge(edge)) {
        return { ...data, hidden: false, labelColor, color: edgeColor }
      }

      const newData = { ...data, hidden: false, labelColor, color: edgeColor }

      if (disableHoverEffect) {
        return newData
      }

      const targetNode = focusedNode || selectedNode
      const edgeHighlightColor = isDarkTheme
        ? Constants.edgeColorHighlightedDarkTheme
        : Constants.edgeColorHighlightedLightTheme

      if (targetNode && graph.hasNode(targetNode)) {
        try {
          const includesNode = graph.extremities(edge).includes(targetNode)
          if (hideUnselectedEdges && !includesNode) {
            newData.hidden = true
          } else if (includesNode) {
            newData.color = edgeHighlightColor
          }
        } catch (error) {
          console.error('Error in edgeReducer:', error)
          return { ...data, hidden: false, labelColor, color: edgeColor }
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
    [sigma, disableHoverEffect]
  )

  /**
   * Keep sigma reducers stable; selection/theme changes are read from refs to avoid
   * re-registering reducers on every hover and maintain frame budget.
   */
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
