import { useRegisterEvents, useSetSettings, useSigma } from '@react-sigma/core'
import { AbstractGraph } from 'graphology-types'
import forceAtlas2 from 'graphology-layout-forceatlas2'
import FA2LayoutSupervisor from 'graphology-layout-forceatlas2/worker'
import { useEffect, useRef } from 'react'

import { EdgeType, NodeType } from '@/hooks/useLightragGraph'
import useIsDarkMode from '@/hooks/useIsDarkMode'
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

  const isDarkTheme = useIsDarkMode()
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

  // The graph the initial FA2 layout has already been run for. Used to run the
  // layout once PER GRAPH, not once per sigma instance (a theme change rebuilds
  // the instance and re-runs the bind effect with the SAME graph).
  const laidOutGraphRef = useRef<unknown>(null)

  /**
   * When component mounts or the graph changes
   * => bind graph to sigma and run the initial layout
   *
   * PERFORMANCE: the initial ForceAtlas2 layout runs in a WEB WORKER with
   * settings inferred from the graph size (inferSettings enables Barnes-Hut
   * above ~2k nodes). The previous implementation called the synchronous
   * `assign()` on the main thread with DEFAULT settings — i.e. without
   * Barnes-Hut, where every iteration is O(V²) pairwise repulsion. At 73k
   * nodes that is billions of operations per iteration, times 15 iterations,
   * on the UI thread: the tab froze for minutes after every load.
   */
  useEffect(() => {
    if (!(sigmaGraph && sigma)) return

    try {
      if (typeof sigma.setGraph === 'function') {
        sigma.setGraph(sigmaGraph as unknown as AbstractGraph<NodeType, EdgeType>)
        console.log('Binding graph to sigma instance')
      } else {
        console.error('Sigma missing setGraph function')
      }
    } catch (error) {
      console.error('Error setting graph on sigma instance:', error)
    }

    if (sigmaGraph.order === 0) return

    // Run the initial layout once PER GRAPH. A theme toggle recreates the sigma
    // instance (settings change -> SigmaContainer rebuild) and re-runs this
    // effect with the SAME graph, which already carries settled positions —
    // re-running FA2 there would replay the relaxation animation on every theme
    // switch. Only (re)bind in that case; skip the layout.
    if (laidOutGraphRef.current === sigmaGraph) return
    laidOutGraphRef.current = sigmaGraph

    let layout: { start: () => void; stop: () => void; kill: () => void } | null = null
    try {
      layout = new FA2LayoutSupervisor(sigmaGraph as never, {
        settings: forceAtlas2.inferSettings(sigmaGraph.order)
      })
      layout.start()
      // Become the single layout owner. If a previous layout is somehow still
      // registered, this kills it so only one supervisor mutates coordinates.
      useGraphStore.getState().setActiveLayoutSupervisor(layout)
      console.log(`FA2 worker layout started (${sigmaGraph.order} nodes)`)
    } catch (error) {
      console.error('Error starting FA2 worker layout:', error)
      return
    }

    // Time budget instead of an iteration count: the UI stays interactive
    // while the layout settles in the background, then we stop it.
    const budgetMs = Constants.workerBudgetMs(sigmaGraph.order)
    const timer = window.setTimeout(() => {
      try {
        layout?.stop()
        console.log('FA2 worker layout stopped after budget')
        // Release the shared slot so the store invariant "activeLayoutSupervisor
        // != null => a layout is running" holds (the budget just stopped this
        // one); no-op for the slot if a manually selected layout already took over.
        useGraphStore.getState().releaseLayoutSupervisor(layout)
        // Clear any stale custom bbox (set by node dragging) and refresh so
        // the camera normalization fits the settled layout.
        sigma.setCustomBBox(null)
        sigma.refresh()
      } catch (error) {
        console.error('Error stopping FA2 worker layout:', error)
      }
    }, budgetMs)

    return () => {
      window.clearTimeout(timer)
      // Release the slot if we still own it (a manually selected worker layout
      // may have taken over and already killed `layout`); otherwise just kill
      // our own supervisor without disturbing the newer layout.
      useGraphStore.getState().releaseLayoutSupervisor(layout)
    }
  }, [sigma, sigmaGraph])

  /**
   * Ensure the sigma instance is set in the store
   */
  useEffect(() => {
    if (sigma) {
      const currentInstance = useGraphStore.getState().sigmaInstance
      if (!currentInstance) {
        console.log('Setting sigma instance from GraphControl')
        useGraphStore.getState().setSigmaInstance(sigma)
      }
    }
  }, [sigma])

  /**
   * With hideEdgesOnMove, edges are skipped while the camera is moving. sigma's
   * built-in post-drag refresh (mouse handleUp) fires on a setTimeout(0), but a
   * drag with inertia is still animating then, so it renders WITHOUT edges and
   * nothing refreshes once movement settles — edges stay hidden until the next
   * interaction (the "edges vanish after a small pan and don't come back" bug;
   * a stage click doesn't help because it isn't a drag, but a hover re-renders).
   *
   * Watch the camera's `updated` event and, once it goes quiet, refresh ONLY
   * when sigma is truly idle. We mirror sigma's exact `moving` condition
   * (camera.isAnimated() || mouse.isMoving || draggedEvents || wheelDirection);
   * if any is still set when the timer fires, we re-arm and wait, so the refresh
   * always lands on a frame where edges are actually drawn.
   */
  useEffect(() => {
    if (!sigma) return
    const camera = sigma.getCamera()
    const mouse = sigma.getMouseCaptor()
    let timer: number | null = null
    const stillMoving = () =>
      camera.isAnimated() ||
      mouse.isMoving ||
      mouse.draggedEvents > 0 ||
      mouse.currentWheelDirection !== 0
    const refreshWhenIdle = () => {
      if (timer !== null) window.clearTimeout(timer)
      timer = window.setTimeout(() => {
        timer = null
        if (stillMoving()) {
          refreshWhenIdle() // not settled yet — keep waiting
          return
        }
        try {
          sigma.refresh()
        } catch {
          /* sigma instance already killed */
        }
      }, 80)
    }
    camera.on('updated', refreshWhenIdle)
    return () => {
      if (timer !== null) window.clearTimeout(timer)
      camera.removeListener('updated', refreshWhenIdle)
    }
  }, [sigma])

  /**
   * When component mounts
   * => register events
   */
  useEffect(() => {
    const { setFocusedNode, setSelectedNode, setFocusedEdge, setSelectedEdge, clearSelection } =
      useGraphStore.getState()

    type NodeEvent = { node: string; event: { original: MouseEvent | TouchEvent } }
    type EdgeEvent = { edge: string; event: { original: MouseEvent | TouchEvent } }

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

    registerEvents(events)
  }, [registerEvents, enableEdgeEvents, sigma])

  /**
   * When edge size settings change, recalculate edge sizes.
   * Batched via updateEachEdgeAttributes (one pass, one graphology event)
   * instead of one event-emitting setEdgeAttribute per edge.
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
      const sizeScale = maxEdgeSize - minEdgeSize
      graph.updateEachEdgeAttributes(
        (_edge, attr) => {
          if (weightRange > 0) {
            const weight = typeof attr.originalWeight === 'number' ? attr.originalWeight : 1
            attr.size = minEdgeSize + sizeScale * Math.pow((weight - minWeight) / weightRange, 0.5)
          } else {
            attr.size = minEdgeSize
          }
          return attr
        },
        { attributes: ['size'] }
      )

      sigma.refresh()
    }
  }, [sigma, sigmaGraph, minEdgeSize, maxEdgeSize])

  /**
   * When focus/selection changes
   * => set the sigma reducers
   *
   * PERFORMANCE:
   * - When nothing is focused or selected, reducers are set to null: running
   *   a JS callback (with an object spread) for all 73k nodes and all edges
   *   on every refresh is pure overhead when there is nothing to highlight.
   * - The focused node's neighborhood is precomputed ONCE into a Set. The
   *   previous code called graph.neighbors(focused).includes(node) INSIDE the
   *   node reducer — allocating the full neighbor array and scanning it
   *   linearly for every single node of the graph, per refresh.
   */
  useEffect(() => {
    const labelColor = isDarkTheme ? Constants.labelColorDarkTheme : undefined
    const edgeColor = isDarkTheme ? Constants.edgeColorDarkTheme : undefined

    const graph = sigma.getGraph()
    const graphOrder = graph ? graph.order : 0
    const effectiveRenderLabels = renderLabels && graphOrder <= Constants.LABEL_RENDER_LIMIT

    const _focusedNode = focusedNode || selectedNode
    const _focusedEdge = focusedEdge || selectedEdge

    if (disableHoverEffect || (!_focusedNode && !_focusedEdge)) {
      setSettings({
        enableEdgeEvents,
        renderEdgeLabels,
        renderLabels: effectiveRenderLabels,
        nodeReducer: null,
        edgeReducer: null
      })
      return
    }

    // Precompute the highlight context once, outside the reducers.
    const neighborSet = new Set<string>()
    let focusedNodeValid = false
    if (_focusedNode && graph.hasNode(_focusedNode)) {
      focusedNodeValid = true
      graph.forEachNeighbor(_focusedNode, (neighbor) => neighborSet.add(neighbor))
    }
    let focusedEdgeSource = ''
    let focusedEdgeTarget = ''
    let focusedEdgeValid = false
    if (!focusedNodeValid && _focusedEdge && graph.hasEdge(_focusedEdge)) {
      focusedEdgeValid = true
      focusedEdgeSource = graph.source(_focusedEdge)
      focusedEdgeTarget = graph.target(_focusedEdge)
    }

    const edgeHighlightColor = isDarkTheme
      ? Constants.edgeColorHighlightedDarkTheme
      : Constants.edgeColorHighlightedLightTheme

    setSettings({
      enableEdgeEvents,
      renderEdgeLabels,
      renderLabels: effectiveRenderLabels,

      nodeReducer: (node, data) => {
        const newData: NodeType & { labelColor?: string; borderColor?: string } = {
          ...data,
          highlighted: false,
          labelColor
        }

        if (focusedNodeValid) {
          if (node === _focusedNode || neighborSet.has(node)) {
            newData.highlighted = true
            if (node === selectedNode) {
              newData.borderColor = Constants.nodeBorderColorSelected
            }
            if (isDarkTheme) {
              newData.labelColor = Constants.LabelColorHighlightedDarkTheme
            }
          } else {
            newData.color = Constants.nodeColorDisabled
          }
        } else if (focusedEdgeValid) {
          if (node === focusedEdgeSource || node === focusedEdgeTarget) {
            newData.highlighted = true
            newData.size = 3
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
        const newData = { ...data, hidden: false, labelColor, color: edgeColor }

        if (focusedNodeValid) {
          // No graph.extremities() here: it allocates an array per edge.
          const touchesFocused =
            graph.source(edge) === _focusedNode || graph.target(edge) === _focusedNode
          if (hideUnselectedEdges) {
            if (!touchesFocused) newData.hidden = true
          } else if (touchesFocused) {
            newData.color = edgeHighlightColor
          }
        } else if (focusedEdgeValid) {
          if (edge === selectedEdge) {
            newData.color = Constants.edgeColorSelected
          } else if (edge === _focusedEdge) {
            newData.color = edgeHighlightColor
          } else if (hideUnselectedEdges) {
            newData.hidden = true
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
    sigmaGraph,
    disableHoverEffect,
    isDarkTheme,
    hideUnselectedEdges,
    enableEdgeEvents,
    renderEdgeLabels,
    renderLabels
  ])

  return null
}

export default GraphControl
