import { useSigma } from '@react-sigma/core'
import { animateNodes } from 'sigma/utils'
import { useLayoutCirclepack } from '@react-sigma/layout-circlepack'
import { useLayoutCircular } from '@react-sigma/layout-circular'
import { LayoutHook, LayoutWorkerHook, WorkerLayoutControlProps } from '@react-sigma/layout-core'
import { useLayoutForce, useWorkerLayoutForce } from '@react-sigma/layout-force'
import { useLayoutForceAtlas2, useWorkerLayoutForceAtlas2 } from '@react-sigma/layout-forceatlas2'
import { useLayoutNoverlap, useWorkerLayoutNoverlap } from '@react-sigma/layout-noverlap'
import { useLayoutRandom } from '@react-sigma/layout-random'
import { useCallback, useMemo, useState, useEffect } from 'react'

import Button from '@/components/ui/Button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import { Command, CommandGroup, CommandItem, CommandList } from '@/components/ui/Command'
import { controlButtonVariant } from '@/lib/constants'
import { useSettingsStore } from '@/stores/settings'

import { GripIcon, PlayIcon, PauseIcon } from 'lucide-react'

type LayoutName =
  | 'Circular'
  | 'Circlepack'
  | 'Random'
  | 'Noverlaps'
  | 'Force Directed'
  | 'Force Atlas'

const WorkerLayoutControl = ({ layout, autoRunFor }: WorkerLayoutControlProps) => {
  const sigma = useSigma()
  const { stop, start, isRunning } = layout

  /**
   * Init component when Sigma or component settings change.
   */
  useEffect(() => {
    if (!sigma) {
      return
    }

    // we run the algo
    let timeout: number | null = null
    if (autoRunFor !== undefined && autoRunFor > -1 && sigma.getGraph().order > 0) {
      start()
      // set a timeout to stop it
      timeout =
        autoRunFor > 0
          ? window.setTimeout(() => { stop() }, autoRunFor) // prettier-ignore
          : null
    }

    //cleaning
    return () => {
      stop()
      if (timeout) {
        clearTimeout(timeout)
      }
    }
  }, [autoRunFor, start, stop, sigma])

  return (
    <Button
      size="icon"
      onClick={() => (isRunning ? stop() : start())}
      tooltip={isRunning ? 'Stop the layout animation' : 'Start the layout animation'}
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
  const [layout, setLayout] = useState<LayoutName>('Circular')
  const [opened, setOpened] = useState<boolean>(false)

  const maxIterations = useSettingsStore.use.graphLayoutMaxIterations()

  const layoutCircular = useLayoutCircular()
  const layoutCirclepack = useLayoutCirclepack()
  const layoutRandom = useLayoutRandom()
  const layoutNoverlap = useLayoutNoverlap({ settings: { margin: 1 } })
  const layoutForce = useLayoutForce({ maxIterations: maxIterations })
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
      console.debug(newLayout)
      const { positions } = layouts[newLayout].layout
      animateNodes(sigma.getGraph(), positions(), { duration: 500 })
      setLayout(newLayout)
    },
    [layouts, sigma]
  )

  return (
    <>
      <div>
        {layouts[layout] && 'worker' in layouts[layout] && (
          <WorkerLayoutControl layout={layouts[layout].worker!} />
        )}
      </div>
      <div>
        <Popover open={opened} onOpenChange={setOpened}>
          <PopoverTrigger asChild>
            <Button
              size="icon"
              variant={controlButtonVariant}
              onClick={() => setOpened((e: boolean) => !e)}
              tooltip="Layout Graph"
            >
              <GripIcon />
            </Button>
          </PopoverTrigger>
          <PopoverContent side="right" align="center" className="p-1">
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
                      {name}
                    </CommandItem>
                  ))}
                </CommandGroup>
              </CommandList>
            </Command>
          </PopoverContent>
        </Popover>
      </div>
    </>
  )
}

export default LayoutsControl
