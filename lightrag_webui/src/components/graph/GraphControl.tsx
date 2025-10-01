import { useRegisterEvents, useSetSettings, useSigma } from '@react-sigma/core'
import { AbstractGraph } from 'graphology-types'
// import { useLayoutCircular } from '@react-sigma/layout-circular'
import { useLayoutForceAtlas2 } from '@react-sigma/layout-forceatlas2'
import { useEffect, useState } from 'react'

// import useRandomGraph, { EdgeType, NodeType } from '@/hooks/useRandomGraph'
import { EdgeType, NodeType } from '@/hooks/useLightragGraph'
import useTheme from '@/hooks/useTheme'
import * as Constants from '@/lib/constants'

import { useSettingsStore } from '@/stores/settings'
import { useGraphStore } from '@/stores/graph'

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
    iterations: maxIterations
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
  const [systemThemeIsDark, setSystemThemeIsDark] = useState(() => 
    window.matchMedia('(prefers-color-scheme: dark)').matches
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
          sigma.setGraph(sigmaGraph as unknown as AbstractGraph<NodeType, EdgeType>);
          console.log('Binding graph to sigma instance');
        } else {
          (sigma as any).graph = sigmaGraph;
          console.warn('Simgma missing setGraph function, set graph property directly');
        }
      } catch (error) {
        console.error('Error setting graph on sigma instance:', error);
      }

      assignLayout();
      console.log('Initial layout applied to graph');
    }
  }, [sigma, sigmaGraph, assignLayout, maxIterations])

  /**
   * Ensure the sigma instance is set in the store
   * This provides a backup in case the instance wasn't set in GraphViewer
   */
  useEffect(() => {
    if (sigma) {
      // Double-check that the store has the sigma instance
      const currentInstance = useGraphStore.getState().sigmaInstance;
      if (!currentInstance) {
        console.log('Setting sigma instance from GraphControl');
        useGraphStore.getState().setSigmaInstance(sigma);
      }
    }
  }, [sigma]);

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
      clickStage: () => clearSelection()
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

      graph.forEachEdge(edge => {
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
        graph.forEachEdge(edge => {
          const weight = graph.getEdgeAttribute(edge, 'originalWeight') || 1
          if (typeof weight === 'number') {
            const scaledSize = minEdgeSize + sizeScale * Math.pow((weight - minWeight) / weightRange, 0.5)
            graph.setEdgeAttribute(edge, 'size', scaledSize)
          }
        })
      } else {
        // If all weights are the same, use default size
        graph.forEachEdge(edge => {
          graph.setEdgeAttribute(edge, 'size', minEdgeSize)
        })
      }

      // Refresh the sigma instance to apply changes
      sigma.refresh()
    }
  }, [sigma, sigmaGraph, minEdgeSize, maxEdgeSize])


  /**
   * When component mount or hovered node change
   * => Setting the sigma reducers
   */
  useEffect(() => {
    // Check if dark mode is actually applied (handles both 'dark' theme and 'system' theme when OS is dark)
    const isDarkTheme = theme === 'dark' || 
      (theme === 'system' && window.document.documentElement.classList.contains('dark'))
    const labelColor = isDarkTheme ? Constants.labelColorDarkTheme : undefined
    const edgeColor = isDarkTheme ? Constants.edgeColorDarkTheme : undefined

    // Update all dynamic settings directly without recreating the sigma container
    setSettings({
      // Update display settings
      enableEdgeEvents,
      renderEdgeLabels,
      renderLabels,

      // Node reducer for node appearance
      nodeReducer: (node, data) => {
        const graph = sigma.getGraph()

        // Add defensive check for node existence during theme switching
        if (!graph.hasNode(node)) {
          console.warn(`Node ${node} not found in graph during theme switch, returning default data`)
          return { ...data, highlighted: false, labelColor }
        }

        const newData: NodeType & {
          labelColor?: string
          borderColor?: string
        } = { ...data, highlighted: data.highlighted || false, labelColor }

        if (!disableHoverEffect) {
          newData.highlighted = false
          const _focusedNode = focusedNode || selectedNode
          const _focusedEdge = focusedEdge || selectedEdge

          if (_focusedNode && graph.hasNode(_focusedNode)) {
            try {
              if (node === _focusedNode || graph.neighbors(_focusedNode).includes(node)) {
                newData.highlighted = true
                if (node === selectedNode) {
                  newData.borderColor = Constants.nodeBorderColorSelected
                }
              }
            } catch (error) {
              console.error('Error in nodeReducer:', error);
              return { ...data, highlighted: false, labelColor }
            }
          } else if (_focusedEdge && graph.hasEdge(_focusedEdge)) {
            try {
              if (graph.extremities(_focusedEdge).includes(node)) {
                newData.highlighted = true
                newData.size = 3
              }
            } catch (error) {
              console.error('Error accessing edge extremities in nodeReducer:', error);
              return { ...data, highlighted: false, labelColor }
            }
          } else {
            return newData
          }

          if (newData.highlighted) {
            if (isDarkTheme) {
              newData.labelColor = Constants.LabelColorHighlightedDarkTheme
            }
          } else {
            newData.color = Constants.nodeColorDisabled
          }
        }
        return newData
      },

      // Edge reducer for edge appearance
      edgeReducer: (edge, data) => {
        const graph = sigma.getGraph()

        // Add defensive check for edge existence during theme switching
        if (!graph.hasEdge(edge)) {
          console.warn(`Edge ${edge} not found in graph during theme switch, returning default data`)
          return { ...data, hidden: false, labelColor, color: edgeColor }
        }

        const newData = { ...data, hidden: false, labelColor, color: edgeColor }

        if (!disableHoverEffect) {
          const _focusedNode = focusedNode || selectedNode

          if (_focusedNode && graph.hasNode(_focusedNode)) {
            try {
              if (hideUnselectedEdges) {
                if (!graph.extremities(edge).includes(_focusedNode)) {
                  newData.hidden = true
                }
              } else {
                if (graph.extremities(edge).includes(_focusedNode)) {
                  newData.color = Constants.edgeColorHighlighted
                }
              }
            } catch (error) {
              console.error('Error in edgeReducer:', error);
              return { ...data, hidden: false, labelColor, color: edgeColor }
            }
          } else {
            const _selectedEdge = selectedEdge && graph.hasEdge(selectedEdge) ? selectedEdge : null;
            const _focusedEdge = focusedEdge && graph.hasEdge(focusedEdge) ? focusedEdge : null;

            if (_selectedEdge || _focusedEdge) {
              if (edge === _selectedEdge) {
                newData.color = Constants.edgeColorSelected
              } else if (edge === _focusedEdge) {
                newData.color = Constants.edgeColorHighlighted
              } else if (hideUnselectedEdges) {
                newData.hidden = true
              }
            }
          }
        }
        return newData
      }
    })
  }, [
    selectedNode,
    focusedNode,
    selectedEdge,
    focusedEdge,
    setSettings,
    sigma,
    disableHoverEffect,
    theme,
    systemThemeIsDark,
    hideUnselectedEdges,
    enableEdgeEvents,
    renderEdgeLabels,
    renderLabels
  ])

  return null
}

export default GraphControl
