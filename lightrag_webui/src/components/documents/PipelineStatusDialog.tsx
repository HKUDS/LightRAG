import { useState, useEffect, useRef, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { X, AlignLeft, AlignCenter, AlignRight } from 'lucide-react'

import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogOverlay,
} from '@/components/ui/AlertDialog'
import Button from '@/components/ui/Button'
import { getPipelineStatus, PipelineStatusResponse } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { cn } from '@/lib/utils'

type DialogPosition = 'left' | 'center' | 'right'

interface PipelineStatusDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export default function PipelineStatusDialog({
  open,
  onOpenChange
}: PipelineStatusDialogProps) {
  const { t } = useTranslation()
  const [status, setStatus] = useState<PipelineStatusResponse | null>(null)
  const [position, setPosition] = useState<DialogPosition>('center')
  const [isUserScrolled, setIsUserScrolled] = useState(false)
  const [historyHeight, setHistoryHeight] = useState('20em')
  const historyRef = useRef<HTMLDivElement>(null)
  const resizeObserverRef = useRef<ResizeObserver | null>(null)

  // Calculate history height based on window height
  const updateHistoryHeight = useCallback(() => {
    const minHeight = 7.5 // 5 lines * 1.5em line height
    const windowHeight = window.innerHeight
    const pixelsPerEm = parseFloat(getComputedStyle(document.documentElement).fontSize)
    const maxHeightInEm = Math.max(Math.floor((windowHeight * 0.4) / pixelsPerEm), minHeight)
    setHistoryHeight(`${maxHeightInEm}em`)
  }, [])

  // Reset position when dialog opens
  useEffect(() => {
    if (open) {
      setPosition('center')
      setIsUserScrolled(false)
      updateHistoryHeight()
    }
  }, [open, updateHistoryHeight])

  // Setup resize observer
  useEffect(() => {
    if (!open) return

    resizeObserverRef.current = new ResizeObserver((entries) => {
      if (entries[0]) {
        updateHistoryHeight()
      }
    })

    resizeObserverRef.current.observe(document.body)

    return () => {
      resizeObserverRef.current?.disconnect()
    }
  }, [open, updateHistoryHeight])

  // Handle scroll position
  useEffect(() => {
    const container = historyRef.current
    if (!container || isUserScrolled) return

    container.scrollTop = container.scrollHeight
  }, [status?.history_messages, isUserScrolled])

  const handleScroll = () => {
    const container = historyRef.current
    if (!container) return

    const isAtBottom = Math.abs(
      (container.scrollHeight - container.scrollTop) - container.clientHeight
    ) < 1
    
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
        toast.error(t('documentPanel.pipelineStatus.errors.fetchFailed', { error: errorMessage(err) }))
      }
    }

    fetchStatus()
    const interval = setInterval(fetchStatus, 2000)
    return () => clearInterval(interval)
  }, [open, t])

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogOverlay className="bg-black/30" />
      <AlertDialogContent 
        className={cn(
          'sm:max-w-[600px] transition-all duration-200',
          position === 'left' && '!left-4 !translate-x-0',
          position === 'center' && '!left-1/2 !-translate-x-1/2',
          position === 'right' && '!right-4 !left-auto !translate-x-0'
        )}
      >
        <AlertDialogHeader className="flex flex-row items-center justify-between">
          <AlertDialogTitle>
            {t('documentPanel.pipelineStatus.title')}
          </AlertDialogTitle>
          
          {/* Position control buttons and close button */}
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                className={cn(
                  'h-6 w-6',
                  position === 'left' && 'bg-zinc-200 text-zinc-800 hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-600'
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
                  position === 'center' && 'bg-zinc-200 text-zinc-800 hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-600'
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
                  position === 'right' && 'bg-zinc-200 text-zinc-800 hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-600'
                )}
                onClick={() => setPosition('right')}
              >
                <AlignRight className="h-4 w-4" />
              </Button>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={() => onOpenChange(false)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </AlertDialogHeader>

        {/* Status Content */}
        <div className="space-y-4 pt-4">
          {/* Pipeline Status */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="text-sm font-medium">{t('documentPanel.pipelineStatus.busy')}:</div>
              <div className={`h-2 w-2 rounded-full ${status?.busy ? 'bg-green-500' : 'bg-gray-300'}`} />
            </div>
            <div className="flex items-center gap-2">
              <div className="text-sm font-medium">{t('documentPanel.pipelineStatus.requestPending')}:</div>
              <div className={`h-2 w-2 rounded-full ${status?.request_pending ? 'bg-green-500' : 'bg-gray-300'}`} />
            </div>
          </div>

          {/* Job Information */}
          <div className="rounded-md border p-3 space-y-2">
            <div>{t('documentPanel.pipelineStatus.jobName')}: {status?.job_name || '-'}</div>
            <div className="flex justify-between">
              <span>{t('documentPanel.pipelineStatus.startTime')}: {status?.job_start ? new Date(status.job_start).toLocaleString() : '-'}</span>
              <span>{t('documentPanel.pipelineStatus.progress')}: {status ? `${status.cur_batch}/${status.batchs}` : '-'}</span>
            </div>
          </div>

          {/* Latest Message */}
          <div className="space-y-2">
            <div className="text-sm font-medium">{t('documentPanel.pipelineStatus.latestMessage')}:</div>
            <div className="font-mono text-sm rounded-md bg-zinc-800 text-zinc-100 p-3">
              {status?.latest_message || '-'}
            </div>
          </div>

          {/* History Messages */}
          <div className="space-y-2">
            <div className="text-sm font-medium">{t('documentPanel.pipelineStatus.historyMessages')}:</div>
            <div 
              ref={historyRef}
              onScroll={handleScroll}
              className="font-mono text-sm rounded-md bg-zinc-800 text-zinc-100 p-3 overflow-y-auto"
              style={{ height: historyHeight }}
            >
              {status?.history_messages?.map((msg, idx) => (
                <div key={idx}>{msg}</div>
              )) || '-'}
            </div>
          </div>
        </div>
      </AlertDialogContent>
    </AlertDialog>
  )
}
