import { useLoadGraph, useRegisterEvents, useSetSettings, useSigma } from '@react-sigma/core'
// import { useLayoutCircular } from '@react-sigma/layout-circular'
import { useLayoutForceAtlas2 } from '@react-sigma/layout-forceatlas2'
import { useEffect } from 'react'

// import useRandomGraph, { EdgeType, NodeType } from '@/hooks/useRandomGraph'
import useLightragGraph, { EdgeType, NodeType } from '@/hooks/useLightragGraph'
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
  const { lightrageGraph } = useLightragGraph()
  const sigma = useSigma<NodeType, EdgeType>()
  const registerEvents = useRegisterEvents<NodeType, EdgeType>()
  const setSettings = useSetSettings<NodeType, EdgeType>()
  const loadGraph = useLoadGraph<NodeType, EdgeType>()

  const maxIterations = useSettingsStore.use.graphLayoutMaxIterations()
  const { assign: assignLayout } = useLayoutForceAtlas2({
    iterations: maxIterations
  })

  const { theme } = useTheme()
  const hideUnselectedEdges = useSettingsStore.use.enableHideUnselectedEdges()
  const selectedNode = useGraphStore.use.selectedNode()
  const focusedNode = useGraphStore.use.focusedNode()
  const selectedEdge = useGraphStore.use.selectedEdge()
  const focusedEdge = useGraphStore.use.focusedEdge()

  /**
   * When component mount or maxIterations changes
   * => load the graph and apply layout
   */
  useEffect(() => {
    // Create & load the graph
    const graph = lightrageGraph()
    loadGraph(graph)
    assignLayout()
  }, [assignLayout, loadGraph, lightrageGraph, maxIterations])

  /**
   * When component mount
   * => register events
   */
  useEffect(() => {
    const { setFocusedNode, setSelectedNode, setFocusedEdge, setSelectedEdge, clearSelection } =
      useGraphStore.getState()

    // Register the events
    registerEvents({
      enterNode: (event) => {
        if (!isButtonPressed(event.event.original)) {
          setFocusedNode(event.node)
        }
      },
      leaveNode: (event) => {
        if (!isButtonPressed(event.event.original)) {
          setFocusedNode(null)
        }
      },
      clickNode: (event) => {
        setSelectedNode(event.node)
        setSelectedEdge(null)
      },
      clickEdge: (event) => {
        setSelectedEdge(event.edge)
        setSelectedNode(null)
      },
      enterEdge: (event) => {
        if (!isButtonPressed(event.event.original)) {
          setFocusedEdge(event.edge)
        }
      },
      leaveEdge: (event) => {
        if (!isButtonPressed(event.event.original)) {
          setFocusedEdge(null)
        }
      },
      clickStage: () => clearSelection()
    })
  }, [registerEvents])

  /**
   * When component mount or hovered node change
   * => Setting the sigma reducers
   */
  useEffect(() => {
    const isDarkTheme = theme === 'dark'
    const labelColor = isDarkTheme ? Constants.labelColorDarkTheme : undefined
    const edgeColor = isDarkTheme ? Constants.edgeColorDarkTheme : undefined

    setSettings({
      nodeReducer: (node, data) => {
        const graph = sigma.getGraph()
        const newData: NodeType & {
          labelColor?: string
          borderColor?: string
        } = { ...data, highlighted: data.highlighted || false, labelColor }

        if (!disableHoverEffect) {
          newData.highlighted = false
          const _focusedNode = focusedNode || selectedNode
          const _focusedEdge = focusedEdge || selectedEdge

          if (_focusedNode) {
            if (node === _focusedNode || graph.neighbors(_focusedNode).includes(node)) {
              newData.highlighted = true
              if (node === selectedNode) {
                newData.borderColor = Constants.nodeBorderColorSelected
              }
            }
          } else if (_focusedEdge) {
            if (graph.extremities(_focusedEdge).includes(node)) {
              newData.highlighted = true
              newData.size = 3
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
      edgeReducer: (edge, data) => {
        const graph = sigma.getGraph()
        const newData = { ...data, hidden: false, labelColor, color: edgeColor }

        if (!disableHoverEffect) {
          const _focusedNode = focusedNode || selectedNode

          if (_focusedNode) {
            if (hideUnselectedEdges) {
              if (!graph.extremities(edge).includes(_focusedNode)) {
                newData.hidden = true
              }
            } else {
              if (graph.extremities(edge).includes(_focusedNode)) {
                newData.color = Constants.edgeColorHighlighted
              }
            }
          } else {
            if (focusedEdge || selectedEdge) {
              if (edge === selectedEdge) {
                newData.color = Constants.edgeColorSelected
              } else if (edge === focusedEdge) {
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
    hideUnselectedEdges
  ])

  return null
}

export default GraphControl
