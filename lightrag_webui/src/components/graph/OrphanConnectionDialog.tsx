import {
  AlertCircle,
  AlignCenter,
  AlignLeft,
  AlignRight,
  CheckCircle2,
  Link,
  Loader2,
  Play,
  Square,
} from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'

import {
  cancelOrphanConnection,
  getOrphanConnectionStatus,
  type OrphanConnectionStatus,
  startOrphanConnection,
} from '@/api/lightrag'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/Dialog'
import Progress from '@/components/ui/Progress'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/Select'
import { cn, errorMessage } from '@/lib/utils'

type DialogPosition = 'left' | 'center' | 'right'

interface OrphanConnectionDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export default function OrphanConnectionDialog({
  open,
  onOpenChange,
}: OrphanConnectionDialogProps) {
  const { t } = useTranslation()
  const [position, setPosition] = useState<DialogPosition>('center')
  const [status, setStatus] = useState<OrphanConnectionStatus | null>(null)
  const [isStarting, setIsStarting] = useState(false)
  const [maxDegree, setMaxDegree] = useState<number>(0)
  const [isCancelling, setIsCancelling] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)

  // Scroll to bottom of messages
  const scrollToBottom = useCallback(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight
    }
  }, [])

  // Poll status every 2 seconds when dialog is open
  useEffect(() => {
    if (!open) return

    const fetchStatus = async () => {
      try {
        const data = await getOrphanConnectionStatus()
        setStatus(data)
      } catch (err) {
        console.error('Failed to fetch orphan connection status:', err)
      }
    }

    // Fetch immediately
    fetchStatus()

    // Then poll every 2 seconds
    const interval = setInterval(fetchStatus, 2000)
    return () => clearInterval(interval)
  }, [open])

  // Auto-scroll when new messages arrive
  useEffect(() => {
    if (status?.history_messages && status.history_messages.length > 0) {
      scrollToBottom()
    }
  }, [status?.history_messages, scrollToBottom])

  // Reset position when dialog opens
  useEffect(() => {
    if (open) {
      setPosition('center')
    }
  }, [open])

  // Handle start
  const handleStart = async () => {
    try {
      setIsStarting(true)
      const result = await startOrphanConnection(3, maxDegree)

      if (result.status === 'already_running') {
        toast.info(t('graphPanel.orphanConnection.alreadyRunning'))
      } else {
        toast.success(t('graphPanel.orphanConnection.started'))
      }
    } catch (err) {
      const errMsg = errorMessage(err)
      toast.error(t('graphPanel.orphanConnection.error', { error: errMsg }))
    } finally {
      setIsStarting(false)
    }
  }

  // Handle cancel
  const handleCancel = async () => {
    try {
      setIsCancelling(true)
      const result = await cancelOrphanConnection()

      if (result.status === 'cancellation_requested') {
        toast.success(t('graphPanel.orphanConnection.cancelSuccess'))
      } else {
        toast.info(t('graphPanel.orphanConnection.notRunning'))
      }
    } catch (err) {
      const errMsg = errorMessage(err)
      toast.error(t('graphPanel.orphanConnection.error', { error: errMsg }))
    } finally {
      setIsCancelling(false)
    }
  }

  // Calculate progress percentage
  const progress = status?.total_orphans
    ? Math.round((status.processed_orphans / status.total_orphans) * 100)
    : 0

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className={cn(
          'sm:max-w-[600px] max-h-[80vh] flex flex-col transition-all duration-200 fixed',
          position === 'left' && '!left-[25%] !translate-x-[-50%] !mx-4',
          position === 'center' && '!left-1/2 !-translate-x-1/2',
          position === 'right' && '!left-[75%] !translate-x-[-50%] !mx-4'
        )}
      >
        <DialogDescription className="sr-only">
          {t('graphPanel.orphanConnection.description')}
        </DialogDescription>
        <DialogHeader className="flex flex-row items-center">
          <DialogTitle className="flex-1">{t('graphPanel.orphanConnection.title')}</DialogTitle>

          {/* Position control buttons */}
          <div className="flex items-center gap-2 mr-8">
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                'h-6 w-6',
                position === 'left' &&
                  'bg-zinc-200 text-zinc-800 hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-600'
              )}
              onClick={() => setPosition('left')}
            >
              <AlignLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                'h-6 w-6',
                position === 'center' &&
                  'bg-zinc-200 text-zinc-800 hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-600'
              )}
              onClick={() => setPosition('center')}
            >
              <AlignCenter className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                'h-6 w-6',
                position === 'right' &&
                  'bg-zinc-200 text-zinc-800 hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-600'
              )}
              onClick={() => setPosition('right')}
            >
              <AlignRight className="h-4 w-4" />
            </Button>
          </div>
        </DialogHeader>

        {/* Content */}
        <div className="flex-1 overflow-hidden flex flex-col space-y-4 pt-4">
          {/* Description */}
          <p className="text-sm text-muted-foreground">
            {t('graphPanel.orphanConnection.explanation')}
          </p>

          {/* Max Degree Selector */}
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium whitespace-nowrap">
              {t('graphPanel.orphanConnection.targetDegree')}:
            </span>
            <Select
              value={maxDegree.toString()}
              onValueChange={(value) => setMaxDegree(Number.parseInt(value, 10))}
              disabled={status?.busy}
            >
              <SelectTrigger className="w-[220px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0">{t('graphPanel.orphanConnection.degree0')}</SelectItem>
                <SelectItem value="1">{t('graphPanel.orphanConnection.degree1')}</SelectItem>
                <SelectItem value="2">{t('graphPanel.orphanConnection.degree2')}</SelectItem>
                <SelectItem value="3">{t('graphPanel.orphanConnection.degree3')}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Status Section */}
          {status && (
            <div className="space-y-3">
              {/* Status indicator */}
              <div
                className={cn(
                  'flex items-center gap-3 p-3 rounded-md border',
                  status.busy
                    ? 'bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800'
                    : status.connections_made > 0
                      ? 'bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800'
                      : 'bg-zinc-50 dark:bg-zinc-900 border-zinc-200 dark:border-zinc-700'
                )}
              >
                {status.busy ? (
                  <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
                ) : status.connections_made > 0 ? (
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                ) : (
                  <Link className="h-5 w-5 text-zinc-500" />
                )}
                <div className="flex-1">
                  <p className="text-sm font-medium">
                    {status.busy
                      ? status.job_name || t('graphPanel.orphanConnection.running')
                      : status.total_orphans > 0
                        ? t('graphPanel.orphanConnection.completed')
                        : t('graphPanel.orphanConnection.ready')}
                  </p>
                  {status.busy && status.cancellation_requested && (
                    <p className="text-xs text-amber-600 dark:text-amber-400">
                      {t('graphPanel.orphanConnection.cancellationPending')}
                    </p>
                  )}
                </div>
              </div>

              {/* Progress bar (only when busy or has results) */}
              {(status.busy || status.total_orphans > 0) && (
                <div className="space-y-2">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>
                      {t('graphPanel.orphanConnection.progress')}: {status.processed_orphans}/
                      {status.total_orphans}
                    </span>
                    <span>{progress}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                </div>
              )}

              {/* Stats */}
              {(status.total_orphans > 0 || status.connections_made > 0) && (
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="p-2 rounded-md bg-zinc-100 dark:bg-zinc-800">
                    <p className="text-lg font-semibold">{status.total_orphans}</p>
                    <p className="text-xs text-muted-foreground">
                      {t('graphPanel.orphanConnection.totalOrphans')}
                    </p>
                  </div>
                  <div className="p-2 rounded-md bg-zinc-100 dark:bg-zinc-800">
                    <p className="text-lg font-semibold">{status.processed_orphans}</p>
                    <p className="text-xs text-muted-foreground">
                      {t('graphPanel.orphanConnection.processed')}
                    </p>
                  </div>
                  <div className="p-2 rounded-md bg-green-100 dark:bg-green-900">
                    <p className="text-lg font-semibold text-green-700 dark:text-green-300">
                      {status.connections_made}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {t('graphPanel.orphanConnection.connectionsMade')}
                    </p>
                  </div>
                </div>
              )}

              {/* Activity Log */}
              {status.history_messages && status.history_messages.length > 0 && (
                <div className="space-y-2">
                  <p className="text-xs font-medium text-muted-foreground">
                    {t('graphPanel.orphanConnection.activityLog')}
                  </p>
                  <div
                    ref={messagesContainerRef}
                    className="h-40 overflow-y-auto rounded-md border bg-zinc-50 dark:bg-zinc-900 p-2 font-mono text-xs"
                  >
                    {status.history_messages.map((msg, idx) => (
                      <div
                        key={`msg-${idx}-${msg.slice(0, 32)}`}
                        className={cn(
                          'py-0.5',
                          msg.includes('Error') && 'text-red-600 dark:text-red-400',
                          msg.includes('Connected:') && 'text-green-600 dark:text-green-400',
                          msg.includes('Completed') &&
                            'text-blue-600 dark:text-blue-400 font-semibold'
                        )}
                      >
                        {msg}
                      </div>
                    ))}
                    <div ref={messagesEndRef} />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex justify-end gap-3 pt-2 border-t">
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              {t('common.close')}
            </Button>

            {status?.busy ? (
              <Button
                variant="destructive"
                onClick={handleCancel}
                disabled={isCancelling || status.cancellation_requested}
                className="min-w-[140px]"
              >
                {isCancelling ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    {t('graphPanel.orphanConnection.cancelling')}
                  </>
                ) : status.cancellation_requested ? (
                  <>
                    <AlertCircle className="h-4 w-4 mr-2" />
                    {t('graphPanel.orphanConnection.cancelling')}
                  </>
                ) : (
                  <>
                    <Square className="h-4 w-4 mr-2" />
                    {t('graphPanel.orphanConnection.cancelButton')}
                  </>
                )}
              </Button>
            ) : (
              <Button onClick={handleStart} disabled={isStarting} className="min-w-[140px]">
                {isStarting ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    {t('graphPanel.orphanConnection.starting')}
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    {t('graphPanel.orphanConnection.startButton')}
                  </>
                )}
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
