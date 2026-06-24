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
  const graphEdgeCount = useGraphStore.use.graphEdgeCount()

  // Mirror GraphViewer's gating: above EDGE_PERF_LIMIT the sigma instance is
  // (re)built without the edge picking buffer, so edge events cannot fire even
  // though the user setting may be on. Use this — not the raw setting — for
  // event registration and the reducers, so we never run the costly edge-focus
  // path against a graph that can't actually pick edges.
  const effectiveEdgeEvents = enableEdgeEvents && graphEdgeCount <= Constants.EDGE_PERF_LIMIT

  // The graph the initial FA2 layout has already been run for. Used to run the
  // layout once PER GRAPH, not once per sigma instance (a theme change rebuilds
  // the instance and re-runs the bind effect with the SAME graph).
  const laidOutGraphRef = useRef<unknown>(null)

  // Last (sigma instance, curved decision) the edge-type effect applied. A
  // rebuild (theme toggle / edge-events gating) creates a fresh sigma whose
  // defaultEdgeType reverts to its construction default ('rect'), so we must
  // re-apply when the INSTANCE changed too — not only when the decision flips.
  const edgeTypeRef = useRef<{ sigma: unknown; curved: boolean } | null>(null)

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
    //
    // The ref is marked "laid out" only when the budget timer fires (layout
    // settled), NOT before start: if a rebuild interrupts the layout mid-run
    // (e.g. edge-events gating crosses the threshold during the first load),
    // the new instance re-runs FA2 from where it was instead of freezing on a
    // half-relaxed layout. Rebuilds after settling still match and skip.
    if (laidOutGraphRef.current === sigmaGraph) return

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
        // Mark this graph as laid out only now (settled), so a rebuild during
        // the budget window re-runs the layout rather than skipping it.
        laidOutGraphRef.current = sigmaGraph
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
      // Update when the instance CHANGED, not only when it's unset. A theme
      // toggle, an effectiveEdgeEvents flip, or crossing the edge threshold
      // rebuilds the SigmaContainer (old instance killed, new one created); if
      // we only wrote on an empty store the killed instance would linger there
      // and consumers (e.g. expand reading sigmaInstance.getCamera()) would act
      // on a dead Sigma.
      if (currentInstance !== sigma) {
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
   * Adapt edge geometry to graph size: curves for small graphs (nicer to read),
   * straight rectangles above EDGE_PERF_LIMIT (curve tessellation is costly).
   *
   * Edges carry no per-edge `type`, so switching `defaultEdgeType` + a full
   * `refresh()` (which re-adds edges through applyEdgeDefaults) re-selects the
   * program for the whole graph without touching attributes or rebuilding.
   *
   * The ref tracks BOTH the sigma instance and the decision: re-apply when the
   * instance changed (a rebuild resets defaultEdgeType to 'rect') OR when the
   * curved/straight decision flipped; skip otherwise so routine expand/prune
   * within one band don't trigger a full refresh.
   */
  useEffect(() => {
    if (!sigma) return
    const curved = graphEdgeCount > 0 && graphEdgeCount <= Constants.EDGE_PERF_LIMIT
    const prev = edgeTypeRef.current
    if (prev && prev.sigma === sigma && prev.curved === curved) return
    edgeTypeRef.current = { sigma, curved }
    setSettings({ defaultEdgeType: curved ? 'curvedNoArrow' : 'rect' })
    try {
      sigma.refresh()
    } catch {
      /* sigma instance already killed */
    }
  }, [sigma, graphEdgeCount, setSettings])

  /**
   * When edge events become gated off (count crossed above EDGE_PERF_LIMIT),
   * drop any residual edge focus/selection so the UI (property panel, reducers)
   * doesn't keep a now-unpickable edge highlighted. Node selection is untouched.
   */
  useEffect(() => {
    if (effectiveEdgeEvents) return
    const { selectedEdge, focusedEdge, setSelectedEdge, setFocusedEdge } = useGraphStore.getState()
    if (selectedEdge !== null) setSelectedEdge(null)
    if (focusedEdge !== null) setFocusedEdge(null)
  }, [effectiveEdgeEvents])

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

    if (effectiveEdgeEvents) {
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
  }, [registerEvents, effectiveEdgeEvents, sigma])

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
    // Ignore any residual edge focus/selection when edge events are gated off
    // (e.g. an edge was selected on a small graph, then expand pushed it past
    // EDGE_PERF_LIMIT): otherwise the edge-focused reducer branch below would
    // still run the per-edge highlight pass on a large graph — exactly the cost
    // we disabled edge events to avoid.
    const _focusedEdge = effectiveEdgeEvents ? focusedEdge || selectedEdge : null

    if (disableHoverEffect || (!_focusedNode && !_focusedEdge)) {
      setSettings({
        enableEdgeEvents: effectiveEdgeEvents,
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
      enableEdgeEvents: effectiveEdgeEvents,
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
    effectiveEdgeEvents,
    renderEdgeLabels,
    renderLabels
  ])

  return null
}

export default GraphControl
