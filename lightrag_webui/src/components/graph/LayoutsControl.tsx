import { useSigma } from '@react-sigma/core'
import { animateNodes } from 'sigma/utils'
import { useLayoutCirclepack } from '@react-sigma/layout-circlepack'
import { useLayoutCircular } from '@react-sigma/layout-circular'
import { LayoutHook, LayoutWorkerHook, WorkerLayoutControlProps } from '@react-sigma/layout-core'
import { useLayoutForce, useWorkerLayoutForce } from '@react-sigma/layout-force'
import { useLayoutForceAtlas2, useWorkerLayoutForceAtlas2 } from '@react-sigma/layout-forceatlas2'
import { useLayoutNoverlap, useWorkerLayoutNoverlap } from '@react-sigma/layout-noverlap'
import { useLayoutRandom } from '@react-sigma/layout-random'
import { useCallback, useMemo, useState, useEffect, useRef } from 'react'

import Button from '@/components/ui/Button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import { Command, CommandGroup, CommandItem, CommandList } from '@/components/ui/Command'
import { controlButtonVariant } from '@/lib/constants'
import { useSettingsStore } from '@/stores/settings'

import { GripIcon, PlayIcon, PauseIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

type LayoutName =
  | 'Circular'
  | 'Circlepack'
  | 'Random'
  | 'Noverlaps'
  | 'Force Directed'
  | 'Force Atlas'

// Extend WorkerLayoutControlProps to include mainLayout
interface ExtendedWorkerLayoutControlProps extends WorkerLayoutControlProps {
  mainLayout: LayoutHook;
}

const WorkerLayoutControl = ({ layout, autoRunFor, mainLayout }: ExtendedWorkerLayoutControlProps) => {
  const sigma = useSigma()
  // Use local state to track animation running status
  const [isRunning, setIsRunning] = useState(false)
  // Timer reference for animation
  const animationTimerRef = useRef<number | null>(null)
  const { t } = useTranslation()

  // Function to update node positions using the layout algorithm
  const updatePositions = useCallback(() => {
    if (!sigma) return

    try {
      const graph = sigma.getGraph()
      if (!graph || graph.order === 0) return

      // Use mainLayout to get positions, similar to refreshLayout function
      // console.log('Getting positions from mainLayout')
      const positions = mainLayout.positions()

      // Animate nodes to new positions
      // console.log('Updating node positions with layout algorithm')
      animateNodes(graph, positions, { duration: 300 }) // Reduced duration for more frequent updates
    } catch (error) {
      console.error('Error updating positions:', error)
      // Stop animation if there's an error
      if (animationTimerRef.current) {
        window.clearInterval(animationTimerRef.current)
        animationTimerRef.current = null
        setIsRunning(false)
      }
    }
  }, [sigma, mainLayout])

  // Improved click handler that uses our own animation timer
  const handleClick = useCallback(() => {
    if (isRunning) {
      // Stop the animation
      console.log('Stopping layout animation')
      if (animationTimerRef.current) {
        window.clearInterval(animationTimerRef.current)
        animationTimerRef.current = null
      }

      // Try to kill the layout algorithm if it's running
      try {
        if (typeof layout.kill === 'function') {
          layout.kill()
          console.log('Layout algorithm killed')
        } else if (typeof layout.stop === 'function') {
          layout.stop()
          console.log('Layout algorithm stopped')
        }
      } catch (error) {
        console.error('Error stopping layout algorithm:', error)
      }

      setIsRunning(false)
    } else {
      // Start the animation
      console.log('Starting layout animation')

      // Initial position update
      updatePositions()

      // Set up interval for continuous updates
      animationTimerRef.current = window.setInterval(() => {
        updatePositions()
      }, 200) // Reduced interval to create overlapping animations for smoother transitions

      setIsRunning(true)

      // Set a timeout to automatically stop the animation after 3 seconds
      setTimeout(() => {
        if (animationTimerRef.current) {
          console.log('Auto-stopping layout animation after 3 seconds')
          window.clearInterval(animationTimerRef.current)
          animationTimerRef.current = null
          setIsRunning(false)

          // Try to stop the layout algorithm
          try {
            if (typeof layout.kill === 'function') {
              layout.kill()
            } else if (typeof layout.stop === 'function') {
              layout.stop()
            }
          } catch (error) {
            console.error('Error stopping layout algorithm:', error)
          }
        }
      }, 3000)
    }
  }, [isRunning, layout, updatePositions])

  /**
   * Init component when Sigma or component settings change.
   */
  useEffect(() => {
    if (!sigma) {
      console.log('No sigma instance available')
      return
    }

    // Auto-run if specified
    let timeout: number | null = null
    if (autoRunFor !== undefined && autoRunFor > -1 && sigma.getGraph().order > 0) {
      console.log('Auto-starting layout animation')

      // Initial position update
      updatePositions()

      // Set up interval for continuous updates
      animationTimerRef.current = window.setInterval(() => {
        updatePositions()
      }, 200) // Reduced interval to create overlapping animations for smoother transitions

      setIsRunning(true)

      // Set a timeout to stop it if autoRunFor > 0
      if (autoRunFor > 0) {
        timeout = window.setTimeout(() => {
          console.log('Auto-stopping layout animation after timeout')
          if (animationTimerRef.current) {
            window.clearInterval(animationTimerRef.current)
            animationTimerRef.current = null
          }
          setIsRunning(false)
        }, autoRunFor)
      }
    }

    // Cleanup function
    return () => {
      // console.log('Cleaning up WorkerLayoutControl')
      if (animationTimerRef.current) {
        window.clearInterval(animationTimerRef.current)
        animationTimerRef.current = null
      }
      if (timeout) {
        window.clearTimeout(timeout)
      }
      setIsRunning(false)
    }
  }, [autoRunFor, sigma, updatePositions])

  return (
    <Button
      size="icon"
      onClick={handleClick}
      tooltip={isRunning ? t('graphPanel.sideBar.layoutsControl.stopAnimation') : t('graphPanel.sideBar.layoutsControl.startAnimation')}
      variant={controlButtonVariant}
    >
      {isRunning ? <PauseIcon /> : <PlayIcon />}
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

  const maxIterations = useSettingsStore.use.graphLayoutMaxIterations()

  const layoutCircular = useLayoutCircular()
  const layoutCirclepack = useLayoutCirclepack()
  const layoutRandom = useLayoutRandom()
  const layoutNoverlap = useLayoutNoverlap({
    maxIterations: maxIterations,
    settings: {
      margin: 5,
      expansion: 1.1,
      gridSize: 1,
      ratio: 1,
      speed: 3,
    }
  })
  // Add parameters for Force Directed layout to improve convergence
  const layoutForce = useLayoutForce({
    maxIterations: maxIterations,
    settings: {
      attraction: 0.0003,  // Lower attraction force to reduce oscillation
      repulsion: 0.02,     // Lower repulsion force to reduce oscillation
      gravity: 0.02,      // Increase gravity to make nodes converge to center faster
      inertia: 0.4,        // Lower inertia to add damping effect
      maxMove: 100         // Limit maximum movement per step to prevent large jumps
    }
  })
  const layoutForceAtlas2 = useLayoutForceAtlas2({ iterations: maxIterations })
  const workerNoverlap = useWorkerLayoutNoverlap()
  const workerForce = useWorkerLayoutForce()
  const workerForceAtlas2 = useWorkerLayoutForceAtlas2()

  const layouts = useMemo(() => {
    return {
      Circular: {
        layout: layoutCircular
      },
      Circlepack: {
        layout: layoutCirclepack
      },
      Random: {
        layout: layoutRandom
      },
      Noverlaps: {
        layout: layoutNoverlap,
        worker: workerNoverlap
      },
      'Force Directed': {
        layout: layoutForce,
        worker: workerForce
      },
      'Force Atlas': {
        layout: layoutForceAtlas2,
        worker: workerForceAtlas2
      }
    } as { [key: string]: { layout: LayoutHook; worker?: LayoutWorkerHook } }
  }, [
    layoutCirclepack,
    layoutCircular,
    layoutForce,
    layoutForceAtlas2,
    layoutNoverlap,
    layoutRandom,
    workerForce,
    workerNoverlap,
    workerForceAtlas2
  ])

  const runLayout = useCallback(
    (newLayout: LayoutName) => {
      console.debug('Running layout:', newLayout)
      const { positions } = layouts[newLayout].layout

      try {
        const graph = sigma.getGraph()
        if (!graph) {
          console.error('No graph available')
          return
        }

        const pos = positions()
        console.log('Positions calculated, animating nodes')
        animateNodes(graph, pos, { duration: 400 })
        setLayout(newLayout)
      } catch (error) {
        console.error('Error running layout:', error)
      }
    },
    [layouts, sigma]
  )

  return (
    <div>
      <div>
        {layouts[layout] && 'worker' in layouts[layout] && (
          <WorkerLayoutControl
            layout={layouts[layout].worker!}
            mainLayout={layouts[layout].layout}
          />
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
            className="p-1 min-w-auto"
          >
            <Command>
              <CommandList>
                <CommandGroup>
                  {Object.keys(layouts).map((name) => (
                    <CommandItem
                      onSelect={() => {
                        runLayout(name as LayoutName)
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
