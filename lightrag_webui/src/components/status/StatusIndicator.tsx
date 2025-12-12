import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import { useBackendState } from '@/stores/state'
import StatusDialog from './StatusDialog'

const StatusIndicator = ({ className }: { className?: string }) => {
  const { t } = useTranslation()
  const health = useBackendState.use.health()
  // Subscribe to trigger re-render when check time updates
  const lastCheckTime = useBackendState.use.lastCheckTime()
  void lastCheckTime
  const status = useBackendState.use.status()
  const [animate, setAnimate] = useState(false)
  const [dialogOpen, setDialogOpen] = useState(false)

  // listen to health change
  useEffect(() => {
    setAnimate(true)
    const timer = setTimeout(() => setAnimate(false), 300)
    return () => clearTimeout(timer)
  }, [])

  return (
    <div className={cn('flex items-center gap-2 opacity-80 select-none', className)}>
      <button
        type="button"
        className="flex cursor-pointer items-center gap-2 bg-transparent border-none p-0"
        onClick={() => setDialogOpen(true)}
      >
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
          {health
            ? t('graphPanel.statusIndicator.connected')
            : t('graphPanel.statusIndicator.disconnected')}
        </span>
      </button>

      <StatusDialog open={dialogOpen} onOpenChange={setDialogOpen} status={status} />
    </div>
  )
}

export default StatusIndicator
