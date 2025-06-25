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

import { XIcon, AlertCircleIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

interface DeselectDocumentsDialogProps {
  selectedCount: number
  onDeselect: () => void
}

export default function DeselectDocumentsDialog({ selectedCount, onDeselect }: DeselectDocumentsDialogProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)

  // Reset state when dialog closes
  useEffect(() => {
    if (!open) {
      // No state to reset for this simple dialog
    }
  }, [open])

  const handleDeselect = useCallback(() => {
    onDeselect()
    setOpen(false)
  }, [onDeselect, setOpen])

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          side="bottom"
          tooltip={t('documentPanel.deselectDocuments.tooltip')}
          size="sm"
        >
          <XIcon/> {t('documentPanel.deselectDocuments.button')}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertCircleIcon className="h-5 w-5" />
            {t('documentPanel.deselectDocuments.title')}
          </DialogTitle>
          <DialogDescription className="pt-2">
            {t('documentPanel.deselectDocuments.description', { count: selectedCount })}
          </DialogDescription>
        </DialogHeader>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            {t('common.cancel')}
          </Button>
          <Button
            variant="default"
            onClick={handleDeselect}
          >
            {t('documentPanel.deselectDocuments.confirmButton')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
