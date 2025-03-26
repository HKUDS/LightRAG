import { useState, useCallback } from 'react'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/Dialog'
import { toast } from 'sonner'
import { errorMessage } from '@/lib/utils'
import { clearDocuments } from '@/api/lightrag'

import { EraserIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

export default function ClearDocumentsDialog() {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)

  const handleClear = useCallback(async () => {
    try {
      const result = await clearDocuments()
      if (result.status === 'success') {
        toast.success(t('documentPanel.clearDocuments.success'))
        setOpen(false)
      } else {
        toast.error(t('documentPanel.clearDocuments.failed', { message: result.message }))
      }
    } catch (err) {
      toast.error(t('documentPanel.clearDocuments.error', { error: errorMessage(err) }))
    }
  }, [setOpen, t])

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" side="bottom" tooltip={t('documentPanel.clearDocuments.tooltip')} size="sm">
          <EraserIcon/> {t('documentPanel.clearDocuments.button')}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>{t('documentPanel.clearDocuments.title')}</DialogTitle>
          <DialogDescription>{t('documentPanel.clearDocuments.confirm')}</DialogDescription>
        </DialogHeader>
        <Button variant="destructive" onClick={handleClear}>
          {t('documentPanel.clearDocuments.confirmButton')}
        </Button>
      </DialogContent>
    </Dialog>
  )
}
