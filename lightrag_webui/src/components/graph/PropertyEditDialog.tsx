import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter
} from '@/components/ui/Dialog'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'

interface PropertyEditDialogProps {
  isOpen: boolean
  onClose: () => void
  onSave: (value: string) => void
  propertyName: string
  initialValue: string
  isSubmitting?: boolean
}

/**
 * Dialog component for editing property values
 * Provides a modal with a title, multi-line text input, and save/cancel buttons
 */
const PropertyEditDialog = ({
  isOpen,
  onClose,
  onSave,
  propertyName,
  initialValue,
  isSubmitting = false
}: PropertyEditDialogProps) => {
  const { t } = useTranslation()
  const [value, setValue] = useState('')

  // Initialize value when dialog opens
  useEffect(() => {
    if (isOpen) {
      setValue(initialValue)
    }
  }, [isOpen, initialValue])

  // Get translated property name
  const getPropertyNameTranslation = (name: string) => {
    const translationKey = `graphPanel.propertiesView.node.propertyNames.${name}`
    const translation = t(translationKey)
    return translation === translationKey ? name : translation
  }

  const handleSave = () => {
    if (value.trim() !== '') {
      onSave(value)
      onClose()
    }
}

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-md" aria-describedby="property-edit-description">
        <DialogHeader>
          <DialogTitle>
            {t('graphPanel.propertiesView.editProperty', {
              property: getPropertyNameTranslation(propertyName)
            })}
          </DialogTitle>
          <p id="property-edit-description" className="text-sm text-muted-foreground">
            {t('graphPanel.propertiesView.editPropertyDescription')}
          </p>
        </DialogHeader>

        {/* Multi-line text input using textarea */}
        <div className="grid gap-4 py-4">
          <textarea
            value={value}
            onChange={(e) => setValue(e.target.value)}
            rows={5}
            className="border-input focus-visible:ring-ring flex w-full rounded-md border bg-transparent px-3 py-2 text-sm shadow-sm transition-colors focus-visible:ring-1 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50"
            disabled={isSubmitting}
          />
        </div>

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={onClose}
            disabled={isSubmitting}
          >
            {t('common.cancel')}
          </Button>
          <Button
            type="button"
            onClick={handleSave}
            disabled={isSubmitting}
          >
            {t('common.save')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default PropertyEditDialog
