import { cn } from '@/lib/utils'
import { useBackendState } from '@/stores/state'
import { useEffect, useState } from 'react'
import StatusDialog from './StatusDialog'
import { useTranslation } from 'react-i18next'

const StatusIndicator = ({ className }: { className?: string }) => {
  const { t } = useTranslation()
  const health = useBackendState.use.health()
  const lastCheckTime = useBackendState.use.lastCheckTime()
  const status = useBackendState.use.status()
  const [animate, setAnimate] = useState(false)
  const [dialogOpen, setDialogOpen] = useState(false)

  // listen to health change
  useEffect(() => {
    setAnimate(true)
    const timer = setTimeout(() => setAnimate(false), 300)
    return () => clearTimeout(timer)
  }, [lastCheckTime])

  return (
    <div className={cn('flex items-center gap-2 opacity-80 select-none', className)}>
      <div
        className="flex cursor-pointer items-center gap-2"
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
          {health ? t('graphPanel.statusIndicator.connected') : t('graphPanel.statusIndicator.disconnected')}
        </span>
      </div>

      <StatusDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        status={status}
      />
    </div>
  )
}

export default StatusIndicator
