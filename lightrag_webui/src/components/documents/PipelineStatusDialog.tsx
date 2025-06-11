import { useState, useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { AlignLeft, AlignCenter, AlignRight } from 'lucide-react'

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription
} from '@/components/ui/Dialog'
import Button from '@/components/ui/Button'
import { getPipelineStatus, PipelineStatusResponse } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { cn } from '@/lib/utils'

type DialogPosition = 'left' | 'center' | 'right'

interface PipelineStatusDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export default function PipelineStatusDialog({ open, onOpenChange }: PipelineStatusDialogProps) {
  const { t } = useTranslation()
  const [status, setStatus] = useState<PipelineStatusResponse | null>(null)
  const [position, setPosition] = useState<DialogPosition>('center')
  const [isUserScrolled, setIsUserScrolled] = useState(false)
  const historyRef = useRef<HTMLDivElement>(null)

  // Reset position when dialog opens
  useEffect(() => {
    if (open) {
      setPosition('center')
      setIsUserScrolled(false)
    }
  }, [open])

  // Handle scroll position
  useEffect(() => {
    const container = historyRef.current
    if (!container || isUserScrolled) return

    container.scrollTop = container.scrollHeight
  }, [status?.history_messages, isUserScrolled])

  const handleScroll = () => {
    const container = historyRef.current
    if (!container) return

    const isAtBottom =
      Math.abs(container.scrollHeight - container.scrollTop - container.clientHeight) < 1

    if (isAtBottom) {
      setIsUserScrolled(false)
    } else {
      setIsUserScrolled(true)
    }
  }

  // Refresh status every 2 seconds
  useEffect(() => {
    if (!open) return

    const fetchStatus = async () => {
      try {
        const data = await getPipelineStatus()
        setStatus(data)
      } catch (err) {
        toast.error(
          t('documentPanel.pipelineStatus.errors.fetchFailed', { error: errorMessage(err) })
        )
      }
    }

    fetchStatus()
    const interval = setInterval(fetchStatus, 2000)
    return () => clearInterval(interval)
  }, [open, t])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className={cn(
          'fixed transition-all duration-200 sm:max-w-[800px]',
          position === 'left' && '!left-[25%] !mx-4 !translate-x-[-50%]',
          position === 'center' && '!left-1/2 !-translate-x-1/2',
          position === 'right' && '!left-[75%] !mx-4 !translate-x-[-50%]'
        )}
      >
        <DialogDescription className="sr-only">
          {status?.job_name
            ? `${t('documentPanel.pipelineStatus.jobName')}: ${status.job_name}, ${t('documentPanel.pipelineStatus.progress')}: ${status.cur_batch}/${status.batchs}`
            : t('documentPanel.pipelineStatus.noActiveJob')}
        </DialogDescription>
        <DialogHeader className="flex flex-row items-center">
          <DialogTitle className="flex-1">{t('documentPanel.pipelineStatus.title')}</DialogTitle>

          {/* Position control buttons */}
          <div className="mr-8 flex items-center gap-2">
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

        {/* Status Content */}
        <div className="space-y-4 pt-4">
          {/* Pipeline Status */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="text-sm font-medium">{t('documentPanel.pipelineStatus.busy')}:</div>
              <div
                className={`h-2 w-2 rounded-full ${status?.busy ? 'bg-green-500' : 'bg-gray-300'}`}
              />
            </div>
            <div className="flex items-center gap-2">
              <div className="text-sm font-medium">
                {t('documentPanel.pipelineStatus.requestPending')}:
              </div>
              <div
                className={`h-2 w-2 rounded-full ${status?.request_pending ? 'bg-green-500' : 'bg-gray-300'}`}
              />
            </div>
          </div>

          {/* Job Information */}
          <div className="space-y-2 rounded-md border p-3">
            <div>
              {t('documentPanel.pipelineStatus.jobName')}: {status?.job_name || '-'}
            </div>
            <div className="flex justify-between">
              <span>
                {t('documentPanel.pipelineStatus.startTime')}:{' '}
                {status?.job_start
                  ? new Date(status.job_start).toLocaleString(undefined, {
                      year: 'numeric',
                      month: 'numeric',
                      day: 'numeric',
                      hour: 'numeric',
                      minute: 'numeric',
                      second: 'numeric'
                    })
                  : '-'}
              </span>
              <span>
                {t('documentPanel.pipelineStatus.progress')}:{' '}
                {status
                  ? `${status.cur_batch}/${status.batchs} ${t('documentPanel.pipelineStatus.unit')}`
                  : '-'}
              </span>
            </div>
          </div>

          {/* Latest Message */}
          <div className="space-y-2">
            <div className="text-sm font-medium">
              {t('documentPanel.pipelineStatus.latestMessage')}:
            </div>
            <div className="rounded-md bg-zinc-800 p-3 font-mono text-xs break-words whitespace-pre-wrap text-zinc-100">
              {status?.latest_message || '-'}
            </div>
          </div>

          {/* History Messages */}
          <div className="space-y-2">
            <div className="text-sm font-medium">
              {t('documentPanel.pipelineStatus.historyMessages')}:
            </div>
            <div
              ref={historyRef}
              onScroll={handleScroll}
              className="max-h-[40vh] min-h-[7.5em] overflow-y-auto rounded-md bg-zinc-800 p-3 font-mono text-xs text-zinc-100"
            >
              {status?.history_messages?.length
                ? status.history_messages.map((msg, idx) => (
                    <div key={idx} className="break-words whitespace-pre-wrap">
                      {msg}
                    </div>
                  ))
                : '-'}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
