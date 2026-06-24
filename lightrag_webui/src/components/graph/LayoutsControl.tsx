import { useSigma } from '@react-sigma/core'
import { animateNodes } from 'sigma/utils'
import { useLayoutCirclepack } from '@react-sigma/layout-circlepack'
import { useLayoutCircular } from '@react-sigma/layout-circular'
import { useLayoutRandom } from '@react-sigma/layout-random'
import { LayoutHook } from '@react-sigma/layout-core'
import forceAtlas2 from 'graphology-layout-forceatlas2'
import FA2Supervisor from 'graphology-layout-forceatlas2/worker'
import NoverlapSupervisor from 'graphology-layout-noverlap/worker'
import ForceSupervisor from 'graphology-layout-force/worker'
import { useCallback, useMemo, useState, useEffect, useRef } from 'react'

import Button from '@/components/ui/Button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import { Command, CommandGroup, CommandItem, CommandList } from '@/components/ui/Command'
import { controlButtonVariant } from '@/lib/constants'
import { useGraphStore } from '@/stores/graph'

import { GripIcon, PlayIcon, PauseIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

type LayoutName =
  | 'Circular'
  | 'Circlepack'
  | 'Random'
  | 'Noverlaps'
  | 'Force Directed'
  | 'Force Atlas'

// The layouts that relax over time and run in a web worker.
type WorkerLayoutName = 'Noverlaps' | 'Force Directed' | 'Force Atlas'
const WORKER_LAYOUTS: ReadonlySet<string> = new Set<WorkerLayoutName>([
  'Noverlaps',
  'Force Directed',
  'Force Atlas'
])

// Above this node count, layout SWITCHES assign positions directly instead of
// animating: animateNodes interpolates every node per frame on the main
// thread, which is far too heavy for large graphs.
const ANIMATE_NODE_LIMIT = 5000

// All three graphology supervisors share this API.
interface LayoutSupervisor {
  start: () => void
  stop: () => void
  kill: () => void
  isRunning: () => boolean
}

const workerBudgetMs = (order: number) => Math.min(1500 + order / 10, 10000)

/**
 * Build a graphology layout supervisor bound to `graph`.
 *
 * IMPORTANT: this binds to the graph that is passed in (the live graph from
 * the store). The @react-sigma worker hooks (useWorkerLayoutForceAtlas2 etc.)
 * construct their supervisor with `sigma.getGraph()` captured in an effect
 * whose dependency list does NOT include the graph, so they stay bound to the
 * empty initial graph that exists at mount. The real graph is swapped in later
 * via `sigma.setGraph()` (in GraphControl), which does not re-run that effect
 * -- so starting those hooks ran the algorithm on a stale empty graph and the
 * visible graph never moved. Building the supervisor here, against the current
 * graph, is what actually makes these layouts work.
 */
const buildSupervisor = (name: WorkerLayoutName, graph: unknown): LayoutSupervisor | null => {
  const order = (graph as { order: number }).order
  switch (name) {
    case 'Force Atlas':
      return new FA2Supervisor(graph as never, {
        settings: forceAtlas2.inferSettings(order)
      }) as unknown as LayoutSupervisor
    case 'Force Directed':
      return new ForceSupervisor(graph as never, {
        settings: {
          attraction: 0.0003,
          repulsion: 0.02,
          gravity: 0.02,
          inertia: 0.4,
          maxMove: 100
        }
      }) as unknown as LayoutSupervisor
    case 'Noverlaps':
      return new NoverlapSupervisor(graph as never, {
        settings: { margin: 5, expansion: 1.1, gridSize: 1, ratio: 1, speed: 3 }
      }) as unknown as LayoutSupervisor
    default:
      return null
  }
}

/**
 * Play/pause control for the worker-backed layouts.
 *
 * Self-contained: it builds and owns the supervisor (bound to the live store
 * graph), auto-runs when its layout is selected, runs for a size-scaled time
 * budget, then stops and reframes the camera. The previous implementation ran
 * the SYNCHRONOUS layout (mainLayout.positions()) every 200ms in a setInterval
 * -- for ForceAtlas2 without Barnes-Hut that is O(V^2) on the main thread,
 * five times a second.
 */
const WorkerLayoutControl = ({ layoutName }: { layoutName: WorkerLayoutName }) => {
  const sigma = useSigma()
  const { t } = useTranslation()
  const [running, setRunning] = useState(false)
  const supervisorRef = useRef<LayoutSupervisor | null>(null)
  const stopTimerRef = useRef<number | null>(null)

  const clearTimer = useCallback(() => {
    if (stopTimerRef.current !== null) {
      window.clearTimeout(stopTimerRef.current)
      stopTimerRef.current = null
    }
  }, [])

  const stop = useCallback(
    (reframe: boolean) => {
      clearTimer()
      try {
        supervisorRef.current?.stop()
      } catch (error) {
        console.error('Error stopping layout:', error)
      }
      setRunning(false)
      // Release the shared slot if we still own it (kills the stopped worker),
      // so "activeLayoutSupervisor != null" reliably means a layout is running.
      const store = useGraphStore.getState()
      if (store.activeLayoutSupervisor === supervisorRef.current) {
        store.setActiveLayoutSupervisor(null)
      }
      if (reframe) {
        try {
          // Clear any custom bbox installed by node dragging, then refit.
          sigma.setCustomBBox(null)
          sigma.getCamera().animatedReset()
        } catch (error) {
          console.error('Error reframing after layout:', error)
        }
      }
    },
    [clearTimer, sigma]
  )

  const start = useCallback(() => {
    const graph = useGraphStore.getState().sigmaGraph
    if (!graph || graph.order === 0) return

    // (Re)build the supervisor bound to the CURRENT graph.
    try {
      supervisorRef.current?.kill()
    } catch {
      /* no live supervisor yet */
    }
    const supervisor = buildSupervisor(layoutName, graph)
    supervisorRef.current = supervisor
    if (!supervisor) return

    // Become the single layout owner. This kills whatever was running before
    // (the initial FA2 from GraphControl, or a previously selected worker
    // layout) so two supervisors never mutate the same coordinates at once.
    useGraphStore.getState().setActiveLayoutSupervisor(supervisor)

    // A custom bbox frozen by dragging breaks coordinate normalization and
    // makes a relaxing layout look like it collapses; clear it before running.
    try {
      sigma.setCustomBBox(null)
    } catch {
      /* ignore */
    }

    // No blocking overlay here: worker layouts are meant to stay interactive
    // while they settle. FA2/Noverlap run in real Web Workers; the play/pause
    // button (driven by `running`) is the live indicator, and the user can
    // pause at any time. The full-screen overlay is reserved for the
    // synchronous switch path in runLayout, which actually blocks the main
    // thread. This also matches the initial FA2 layout in GraphControl, which
    // deliberately does not set isLayoutComputing.
    //
    // We still DEFER supervisor.start() so the browser paints the Pause state
    // first. Force's runFrame() runs sync on the main thread; rAF won't help
    // because rAF callbacks fire BEFORE the next paint, so a small setTimeout
    // guarantees a paint lands first. FA2/Noverlap don't strictly need this
    // (their start() just posts to a real worker and returns), but the small
    // delay is irrelevant for them too.
    setRunning(true)
    window.setTimeout(() => {
      // Bail if the user already switched layouts before we fired.
      if (supervisorRef.current !== supervisor) return
      try {
        supervisor.start()
      } catch (error) {
        console.error('Error starting layout:', error)
        setRunning(false)
        return
      }
      clearTimer()
      stopTimerRef.current = window.setTimeout(() => stop(true), workerBudgetMs(graph.order))
    }, 50)
  }, [layoutName, sigma, clearTimer, stop])

  // Auto-run when this worker layout becomes active; clean up on
  // change/unmount. Keyed on layoutName only: supervisorRef is updated INSIDE
  // start() (which runs after this effect's cleanup), so cleanup correctly
  // kills the PREVIOUS layout's supervisor rather than the incoming one.
  useEffect(() => {
    // Intentional: mounting this control means the layout was selected, so we
    // auto-run it (start() spins up the worker supervisor and flips `running`).
    // This is the "synchronize with an external system" case the rule exempts,
    // not an accidental render cascade.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    start()
    return () => {
      clearTimer()
      // Release the shared slot if we still own it; otherwise just kill our
      // own (previous) supervisor. start() for the incoming layout runs AFTER
      // this cleanup, so the slot still points at our supervisor here.
      const store = useGraphStore.getState()
      if (store.activeLayoutSupervisor === supervisorRef.current) {
        store.setActiveLayoutSupervisor(null) // kills our supervisor
      } else {
        try {
          supervisorRef.current?.kill()
        } catch {
          /* already dead */
        }
      }
      supervisorRef.current = null
      setRunning(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layoutName])

  return (
    <Button
      size="icon"
      onClick={() => (running ? stop(true) : start())}
      tooltip={
        running
          ? t('graphPanel.sideBar.layoutsControl.stopAnimation')
          : t('graphPanel.sideBar.layoutsControl.startAnimation')
      }
      variant={controlButtonVariant}
    >
      {running ? <PauseIcon /> : <PlayIcon />}
    </Button>
  )
}

/**
 * Component that controls the layout of the graph.
 */
const LayoutsControl = () => {
  const sigma = useSigma()
  const { t } = useTranslation()
  const [layout, setLayout] = useState<LayoutName>('Circular')
  const [opened, setOpened] = useState<boolean>(false)

  // Only the instant, deterministic layouts use the @react-sigma hooks (they
  // compute positions synchronously and we assign them). The relaxing layouts
  // are handled by WorkerLayoutControl via supervisors.
  const layoutCircular = useLayoutCircular()
  const layoutCirclepack = useLayoutCirclepack()
  const layoutRandom = useLayoutRandom()

  const syncLayouts = useMemo(() => {
    return {
      Circular: layoutCircular,
      Circlepack: layoutCirclepack,
      Random: layoutRandom
    } as Record<string, LayoutHook>
  }, [layoutCircular, layoutCirclepack, layoutRandom])

  const allLayoutNames: LayoutName[] = useMemo(
    () => ['Circular', 'Circlepack', 'Random', 'Noverlaps', 'Force Directed', 'Force Atlas'],
    []
  )

  const runLayout = useCallback(
    (newLayout: LayoutName) => {
      console.debug('Running layout:', newLayout)

      // Worker layouts: selecting one (re)mounts WorkerLayoutControl, which
      // auto-runs it against the live graph. Nothing is computed on the main
      // thread here.
      if (WORKER_LAYOUTS.has(newLayout)) {
        setLayout(newLayout)
        return
      }

      const graph = sigma.getGraph()
      if (!graph || graph.order === 0) {
        console.error('No graph available')
        return
      }

      // BUGFIX "graph collapses into a single point": node dragging installs
      // a custom bounding box that was never cleared, freezing sigma's
      // coordinate normalization to the previous layout's extent.
      const doLayout = () => {
        try {
          // Kill any running worker layout (notably the initial FA2, which has
          // no WorkerLayoutControl to unmount) before assigning positions, or
          // it keeps mutating coordinates and fights this synchronous layout.
          useGraphStore.getState().setActiveLayoutSupervisor(null)
          const pos = syncLayouts[newLayout].positions()
          sigma.setCustomBBox(null)
          if (graph.order > ANIMATE_NODE_LIMIT) {
            graph.updateEachNodeAttributes(
              (node, attr) => {
                const p = pos[node]
                if (p) {
                  attr.x = p.x
                  attr.y = p.y
                }
                return attr
              },
              { attributes: ['x', 'y'] }
            )
            sigma.refresh()
          } else {
            animateNodes(graph, pos, { duration: 400 })
          }
          sigma.getCamera().animatedReset()
          setLayout(newLayout)
        } catch (error) {
          console.error('Error running layout:', error)
        }
      }

      // For large graphs, the assign + refresh blocks the main thread for
      // hundreds of ms. Show the loading overlay and defer the heavy work
      // one frame so React actually paints the overlay before we block.
      // Small graphs animate (400ms) -- the animation itself is feedback.
      if (graph.order > ANIMATE_NODE_LIMIT) {
        useGraphStore.getState().setIsLayoutComputing(true)
        window.requestAnimationFrame(() => {
          try {
            doLayout()
          } finally {
            useGraphStore.getState().setIsLayoutComputing(false)
          }
        })
      } else {
        doLayout()
      }
    },
    [syncLayouts, sigma]
  )

  return (
    <div>
      <div>
        {WORKER_LAYOUTS.has(layout) && (
          <WorkerLayoutControl layoutName={layout as WorkerLayoutName} />
        )}
      </div>
      <div>
        <Popover open={opened} onOpenChange={setOpened}>
          <PopoverTrigger asChild>
            <Button
              size="icon"
              variant={controlButtonVariant}
              onClick={() => setOpened((e: boolean) => !e)}
              tooltip={t('graphPanel.sideBar.layoutsControl.layoutGraph')}
            >
              <GripIcon />
            </Button>
          </PopoverTrigger>
          <PopoverContent
            side="right"
            align="start"
            sideOffset={8}
            collisionPadding={5}
            sticky="always"
            className="min-w-auto p-1"
          >
            <Command>
              <CommandList>
                <CommandGroup>
                  {allLayoutNames.map((name) => (
                    <CommandItem
                      onSelect={() => {
                        runLayout(name)
                      }}
                      key={name}
                      className="cursor-pointer text-xs"
                    >
                      {t(`graphPanel.sideBar.layoutsControl.layouts.${name}`)}
                    </CommandItem>
                  ))}
                </CommandGroup>
              </CommandList>
            </Command>
          </PopoverContent>
        </Popover>
      </div>
    </div>
  )
}

export default LayoutsControl
