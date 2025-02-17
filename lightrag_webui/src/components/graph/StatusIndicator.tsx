import { cn } from '@/lib/utils'
import { useBackendState } from '@/stores/state'
import { useEffect, useState } from 'react'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import StatusCard from '@/components/graph/StatusCard'

const StatusIndicator = () => {
  const health = useBackendState.use.health()
  const lastCheckTime = useBackendState.use.lastCheckTime()
  const status = useBackendState.use.status()
  const [animate, setAnimate] = useState(false)

  // listen to health change
  useEffect(() => {
    setAnimate(true)
    const timer = setTimeout(() => setAnimate(false), 300)
    return () => clearTimeout(timer)
  }, [lastCheckTime])

  return (
    <div className="fixed right-4 bottom-4 flex items-center gap-2 opacity-80 select-none">
      <Popover>
        <PopoverTrigger asChild>
          <div className="flex cursor-help items-center gap-2">
            <div
              className={cn(
                'h-3 w-3 rounded-full transition-all duration-300',
                'shadow-[0_0_8px_rgba(0,0,0,0.2)]',
                health ? 'bg-green-500' : 'bg-red-500',
                animate && 'scale-125',
                animate && health && 'shadow-[0_0_12px_rgba(34,197,94,0.4)]',
                animate && !health && 'shadow-[0_0_12px_rgba(239,68,68,0.4)]'
              )}
            />
            <span className="text-muted-foreground text-xs">
              {health ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </PopoverTrigger>
        <PopoverContent className="w-auto" side="top" align="end">
          <StatusCard status={status} />
        </PopoverContent>
      </Popover>
    </div>
  )
}

export default StatusIndicator
