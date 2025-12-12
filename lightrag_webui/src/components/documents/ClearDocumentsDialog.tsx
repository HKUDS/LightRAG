import { AlertTriangleIcon, EraserIcon, Loader2Icon } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { clearCache, clearDocuments } from '@/api/lightrag'
import Button from '@/components/ui/Button'
import Checkbox from '@/components/ui/Checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/Dialog'
import Input from '@/components/ui/Input'
import { errorMessage } from '@/lib/utils'

// Simple Label component
const Label = ({
  htmlFor,
  className,
  children,
  ...props
}: React.LabelHTMLAttributes<HTMLLabelElement>) => (
  <label htmlFor={htmlFor} className={className} {...props}>
    {children}
  </label>
)

interface ClearDocumentsDialogProps {
  onDocumentsCleared?: () => Promise<void>
}

export default function ClearDocumentsDialog({ onDocumentsCleared }: ClearDocumentsDialogProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [confirmText, setConfirmText] = useState('')
  const [clearCacheOption, setClearCacheOption] = useState(false)
  const [isClearing, setIsClearing] = useState(false)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isConfirmEnabled = confirmText.toLowerCase() === 'yes'

  // Timeout constant (30 seconds)
  const CLEAR_TIMEOUT = 30000

  // Reset state when dialog closes
  useEffect(() => {
    if (!open) {
      setConfirmText('')
      setClearCacheOption(false)
      setIsClearing(false)

      // Clear timeout timer
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
        timeoutRef.current = null
      }
    }
  }, [open])

  // Cleanup when component unmounts
  useEffect(() => {
    return () => {
      // Clear timeout timer when component unmounts
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [])

  const handleClear = useCallback(async () => {
    if (!isConfirmEnabled || isClearing) return

    setIsClearing(true)

    // Set timeout protection
    timeoutRef.current = setTimeout(() => {
      if (isClearing) {
        toast.error(t('documentPanel.clearDocuments.timeout'))
        setIsClearing(false)
        setConfirmText('') // Reset confirmation text after timeout
      }
    }, CLEAR_TIMEOUT)

    try {
      const result = await clearDocuments()

      if (result.status !== 'success') {
        toast.error(t('documentPanel.clearDocuments.failed', { message: result.message }))
        setConfirmText('')
        return
      }

      toast.success(t('documentPanel.clearDocuments.success'))

      if (clearCacheOption) {
        try {
          await clearCache()
          toast.success(t('documentPanel.clearDocuments.cacheCleared'))
        } catch (cacheErr) {
          toast.error(
            t('documentPanel.clearDocuments.cacheClearFailed', { error: errorMessage(cacheErr) })
          )
        }
      }

      // Refresh document list if provided
      if (onDocumentsCleared) {
        onDocumentsCleared().catch(console.error)
      }

      // Close dialog after all operations succeed
      setOpen(false)
    } catch (err) {
      toast.error(t('documentPanel.clearDocuments.error', { error: errorMessage(err) }))
      setConfirmText('')
    } finally {
      // Clear timeout timer
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
        timeoutRef.current = null
      }
      setIsClearing(false)
    }
  }, [isConfirmEnabled, isClearing, clearCacheOption, t, onDocumentsCleared])

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          side="bottom"
          tooltip={t('documentPanel.clearDocuments.tooltip')}
          size="sm"
        >
          <EraserIcon /> {t('documentPanel.clearDocuments.button')}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-red-500 dark:text-red-400 font-bold">
            <AlertTriangleIcon className="h-5 w-5" />
            {t('documentPanel.clearDocuments.title')}
          </DialogTitle>
          <DialogDescription className="pt-2">
            {t('documentPanel.clearDocuments.description')}
          </DialogDescription>
        </DialogHeader>

        <div className="text-red-500 dark:text-red-400 font-semibold mb-4">
          {t('documentPanel.clearDocuments.warning')}
        </div>
        <div className="mb-4">{t('documentPanel.clearDocuments.confirm')}</div>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="confirm-text" className="text-sm font-medium">
              {t('documentPanel.clearDocuments.confirmPrompt')}
            </Label>
            <Input
              id="confirm-text"
              value={confirmText}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setConfirmText(e.target.value)}
              placeholder={t('documentPanel.clearDocuments.confirmPlaceholder')}
              className="w-full"
              disabled={isClearing}
            />
          </div>

          <div className="flex items-center space-x-2">
            <Checkbox
              id="clear-cache"
              checked={clearCacheOption}
              onCheckedChange={(checked: boolean | 'indeterminate') =>
                setClearCacheOption(checked === true)
              }
              disabled={isClearing}
            />
            <Label htmlFor="clear-cache" className="text-sm font-medium cursor-pointer">
              {t('documentPanel.clearDocuments.clearCache')}
            </Label>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)} disabled={isClearing}>
            {t('common.cancel')}
          </Button>
          <Button
            variant="destructive"
            onClick={handleClear}
            disabled={!isConfirmEnabled || isClearing}
          >
            {isClearing ? (
              <>
                <Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
                {t('documentPanel.clearDocuments.clearing')}
              </>
            ) : (
              t('documentPanel.clearDocuments.confirmButton')
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
