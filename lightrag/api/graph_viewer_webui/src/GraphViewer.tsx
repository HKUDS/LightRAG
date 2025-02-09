import { useEffect, useState, useCallback } from 'react'
// import { MiniMap } from '@react-sigma/minimap'
import { SigmaContainer, useRegisterEvents, useSigma } from '@react-sigma/core'
import { Settings as SigmaSettings } from 'sigma/settings'
import { GraphSearchOption } from '@react-sigma/graph-search'
import { EdgeArrowProgram, NodePointProgram, NodeCircleProgram } from 'sigma/rendering'
import { NodeBorderProgram } from '@sigma/node-border'
import EdgeCurveProgram, { EdgeCurvedArrowProgram } from '@sigma/edge-curve'

import FocusOnNode from '@/components/FocusOnNode'
import LayoutsControl from '@/components/LayoutsControl'
import GraphControl from '@/components/GraphControl'
import ThemeToggle from '@/components/ThemeToggle'
import ZoomControl from '@/components/ZoomControl'
import FullScreenControl from '@/components/FullScreenControl'
import Settings from '@/components/Settings'
import GraphSearch from '@/components/GraphSearch'

import { useSettingsStore } from '@/lib/settings'

import '@react-sigma/core/lib/style.css'
import '@react-sigma/graph-search/lib/style.css'

// Sigma settings
const defaultSigmaSettings: Partial<SigmaSettings> = {
  allowInvalidContainer: true,
  defaultNodeType: 'default',
  defaultEdgeType: 'curvedArrow',
  renderEdgeLabels: false,
  edgeProgramClasses: {
    arrow: EdgeArrowProgram,
    curvedArrow: EdgeCurvedArrowProgram,
    curvedNoArrow: EdgeCurveProgram
  },
  nodeProgramClasses: {
    default: NodeBorderProgram,
    circel: NodeCircleProgram,
    point: NodePointProgram
  },
  labelGridCellSize: 60,
  labelRenderedSizeThreshold: 12,
  enableEdgeEvents: true,
  labelColor: {
    color: '#000',
    attribute: 'labelColor'
  },
  edgeLabelColor: {
    color: '#000',
    attribute: 'labelColor'
  },
  edgeLabelSize: 8,
  labelSize: 12
  // minEdgeThickness: 2
  // labelFont: 'Lato, sans-serif'
}

const GraphEvents = () => {
  const registerEvents = useRegisterEvents()
  const sigma = useSigma()
  const [draggedNode, setDraggedNode] = useState<string | null>(null)

  useEffect(() => {
    // Register the events
    registerEvents({
      downNode: (e) => {
        setDraggedNode(e.node)
        sigma.getGraph().setNodeAttribute(e.node, 'highlighted', true)
      },
      // On mouse move, if the drag mode is enabled, we change the position of the draggedNode
      mousemovebody: (e) => {
        if (!draggedNode) return
        // Get new position of node
        const pos = sigma.viewportToGraph(e)
        sigma.getGraph().setNodeAttribute(draggedNode, 'x', pos.x)
        sigma.getGraph().setNodeAttribute(draggedNode, 'y', pos.y)

        // Prevent sigma to move camera:
        e.preventSigmaDefault()
        e.original.preventDefault()
        e.original.stopPropagation()
      },
      // On mouse up, we reset the autoscale and the dragging mode
      mouseup: () => {
        if (draggedNode) {
          setDraggedNode(null)
          sigma.getGraph().removeNodeAttribute(draggedNode, 'highlighted')
        }
      },
      // Disable the autoscale at the first down interaction
      mousedown: () => {
        if (!sigma.getCustomBBox()) sigma.setCustomBBox(sigma.getBBox())
      }
    })
  }, [registerEvents, sigma, draggedNode])

  return null
}

export const GraphViewer = () => {
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [focusedNode, setFocusedNode] = useState<string | null>(null)
  const [sigmaSettings, setSigmaSettings] = useState(defaultSigmaSettings)
  const [autoMoveToFocused, setAutoMoveToFocused] = useState(false)

  const enableEdgeEvents = useSettingsStore.use.enableEdgeEvents()
  const enableNodeDrag = useSettingsStore.use.enableNodeDrag()
  const renderEdgeLabels = useSettingsStore.use.showEdgeLabel()

  useEffect(() => {
    setSigmaSettings({
      ...defaultSigmaSettings,
      enableEdgeEvents,
      renderEdgeLabels
    })
  }, [enableEdgeEvents, renderEdgeLabels])

  const onFocus = useCallback(
    (value: GraphSearchOption | null) => {
      if (value === null) setFocusedNode(null)
      else if (value.type === 'nodes') setFocusedNode(value.id)
    },
    [setFocusedNode]
  )

  const onSelect = useCallback(
    (value: GraphSearchOption | null) => {
      if (value === null) setSelectedNode(null)
      else if (value.type === 'nodes') {
        setAutoMoveToFocused(true)
        setSelectedNode(value.id)
        setTimeout(() => setAutoMoveToFocused(false), 100)
      }
    },
    [setSelectedNode, setAutoMoveToFocused]
  )

  return (
    <SigmaContainer settings={sigmaSettings} className="!bg-background !size-full overflow-hidden">
      <GraphControl
        selectedNode={selectedNode}
        setSelectedNode={setSelectedNode}
        focusedNode={focusedNode}
        setFocusedNode={setFocusedNode}
      />

      {enableNodeDrag && <GraphEvents />}

      <FocusOnNode node={focusedNode ?? selectedNode} move={autoMoveToFocused} />

      <div className="absolute top-2 right-2">
        <GraphSearch
          type="nodes"
          value={selectedNode ? { type: 'nodes', id: selectedNode } : null}
          onFocus={onFocus}
          onChange={onSelect}
        />
      </div>

      <div className="bg-background/20 absolute bottom-2 left-2 flex flex-col rounded-xl border-2 backdrop-blur-lg">
        <Settings />
        <ZoomControl />
        <LayoutsControl />
        <FullScreenControl />
        <ThemeToggle />
      </div>

      {/* <div className="absolute bottom-2 right-2 flex flex-col rounded-xl border-2">
        <MiniMap width="100px" height="100px" />
      </div> */}
    </SigmaContainer>
  )
}
