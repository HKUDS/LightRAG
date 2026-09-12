import { useState, useCallback, useEffect, useRef } from 'react'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter
} from '@/components/ui/Dialog'
import Input from '@/components/ui/Input'
import Checkbox from '@/components/ui/Checkbox'
import { toast } from 'sonner'
import { errorMessage } from '@/lib/utils'
import { clearDocuments } from '@/api/lightrag'

import { EraserIcon, AlertTriangleIcon, Loader2Icon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

// Simple Label component
const Label = ({
  htmlFor,
  className,
  children,
  ...props
}: React.LabelHTMLAttributes<HTMLLabelElement>) => (
  <label
    htmlFor={htmlFor}
    className={className}
    {...props}
  >
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

  // Reset state when dialog closes - handled in onOpenChange to avoid setState in effect
  const handleOpenChange = useCallback((newOpen: boolean) => {
    setOpen(newOpen)
    if (!newOpen) {
      setConfirmText('')
      setClearCacheOption(false)
      setIsClearing(false)

      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
        timeoutRef.current = null
      }
    }
  }, [])

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
      // The cache drop rides along on the clear request itself so it runs
      // inside the server's destructive reservation; there is no longer a
      // separate endpoint that clears it without one.
      const result = await clearDocuments(clearCacheOption)

      // `partial_success` means the documents WERE cleared -- some storage,
      // some input file or the opted-in cache drop reported an error, and the
      // server names them in `message`. Only `busy` (nothing ran) and `fail`
      // (every storage drop failed) leave the documents in place. Treating
      // partial_success as a failure would keep the dialog open over a stale
      // list, telling the user the documents are still there when they are
      // not -- so it refreshes and closes like a success, with the errors
      // surfaced as a warning instead of a success toast.
      if (result.status !== 'success' && result.status !== 'partial_success') {
        toast.error(t('documentPanel.clearDocuments.failed', { message: result.message }))
        setConfirmText('')
        return
      }

      if (result.status === 'partial_success') {
        toast.warning(
          t('documentPanel.clearDocuments.partialSuccess', { message: result.message })
        )
      } else {
        toast.success(
          clearCacheOption
            ? t('documentPanel.clearDocuments.successWithCache')
            : t('documentPanel.clearDocuments.success')
        )
      }

      // Refresh document list if provided
      if (onDocumentsCleared) {
        onDocumentsCleared().catch(console.error)
      }

      // Close dialog after all operations succeed
      handleOpenChange(false)
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
  }, [isConfirmEnabled, isClearing, clearCacheOption, handleOpenChange, t, onDocumentsCleared, CLEAR_TIMEOUT])

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button variant="outline" side="bottom" tooltip={t('documentPanel.clearDocuments.tooltip')} size="sm">
          <EraserIcon/> {t('documentPanel.clearDocuments.button')}
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
        <div className="mb-4">
          {t('documentPanel.clearDocuments.confirm')}
        </div>

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
              onCheckedChange={(checked: boolean | 'indeterminate') => setClearCacheOption(checked === true)}
              disabled={isClearing}
            />
            <Label htmlFor="clear-cache" className="text-sm font-medium cursor-pointer">
              {t('documentPanel.clearDocuments.clearCache')}
            </Label>
          </div>
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => handleOpenChange(false)}
            disabled={isClearing}
          >
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
