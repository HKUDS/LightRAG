import { useState, useCallback, useEffect } from 'react'
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
import { clearDocuments, clearCache } from '@/api/lightrag'
import { useBackendState } from '@/stores/state'

import { EraserIcon, AlertTriangleIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

// 简单的Label组件
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
  const isConfirmEnabled = confirmText.toLowerCase() === 'yes'
  const check = useBackendState.use.check()

  // 重置状态当对话框关闭时
  useEffect(() => {
    if (!open) {
      setConfirmText('')
      setClearCacheOption(false)
    }
  }, [open])

  const handleClear = useCallback(async () => {
    if (!isConfirmEnabled) return

    try {
      const result = await clearDocuments()

      if (clearCacheOption) {
        try {
          await clearCache()
          toast.success(t('documentPanel.clearDocuments.cacheCleared'))
        } catch (cacheErr) {
          toast.error(t('documentPanel.clearDocuments.cacheClearFailed', { error: errorMessage(cacheErr) }))
        }
      }

      if (result.status === 'success') {
        toast.success(t('documentPanel.clearDocuments.success'))
      } else {
        toast.error(t('documentPanel.clearDocuments.failed', { message: result.message }))
      }
    } catch (err) {
      toast.error(t('documentPanel.clearDocuments.error', { error: errorMessage(err) }))
    } finally {
      // Execute these operations regardless of success or failure
      try {
        // Update backend state
        await check()

        // Refresh document list
        if (onDocumentsCleared) {
          await onDocumentsCleared()
        }
      } catch (refreshErr) {
        console.error('Error refreshing state:', refreshErr)
      }

      setOpen(false)
    }
  }, [isConfirmEnabled, clearCacheOption, setOpen, t, check, onDocumentsCleared])

  return (
    <Dialog open={open} onOpenChange={setOpen}>
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
            <div className="text-red-500 dark:text-red-400 font-semibold mb-4">
              {t('documentPanel.clearDocuments.warning')}
            </div>
            <div className="mb-4">
              {t('documentPanel.clearDocuments.confirm')}
            </div>
          </DialogDescription>
        </DialogHeader>

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
            />
          </div>

          <div className="flex items-center space-x-2">
            <Checkbox
              id="clear-cache"
              checked={clearCacheOption}
              onCheckedChange={(checked: boolean | 'indeterminate') => setClearCacheOption(checked === true)}
            />
            <Label htmlFor="clear-cache" className="text-sm font-medium cursor-pointer">
              {t('documentPanel.clearDocuments.clearCache')}
            </Label>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            {t('common.cancel')}
          </Button>
          <Button
            variant="destructive"
            onClick={handleClear}
            disabled={!isConfirmEnabled}
          >
            {t('documentPanel.clearDocuments.confirmButton')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
