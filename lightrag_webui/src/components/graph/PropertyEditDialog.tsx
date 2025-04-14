import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription
} from '@/components/ui/Dialog'
import Button from '@/components/ui/Button'

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

  // Get textarea configuration based on property name
  const getTextareaConfig = (propertyName: string) => {
    switch (propertyName) {
    case 'description':
      return {
        // No rows attribute for description to allow auto-sizing
        className: 'max-h-[50vh] min-h-[10em] resize-y', // Maximum height 70% of viewport, minimum height ~20 lines, allow vertical resizing
        style: {
          height: '70vh', // Set initial height to 70% of viewport
          minHeight: '20em', // Minimum height ~20 lines
          resize: 'vertical' as const // Allow vertical resizing, using 'as const' to fix type
        }
      };
    case 'entity_id':
      return {
        rows: 2,
        className: '',
        style: {}
      };
    case 'keywords':
      return {
        rows: 4,
        className: '',
        style: {}
      };
    default:
      return {
        rows: 5,
        className: '',
        style: {}
      };
    }
  };

  const handleSave = () => {
    if (value.trim() !== '') {
      onSave(value)
      onClose()
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {t('graphPanel.propertiesView.editProperty', {
              property: getPropertyNameTranslation(propertyName)
            })}
          </DialogTitle>
          <DialogDescription>
            {t('graphPanel.propertiesView.editPropertyDescription')}
          </DialogDescription>
        </DialogHeader>

        {/* Multi-line text input using textarea */}
        <div className="grid gap-4 py-4">
          {(() => {
            const config = getTextareaConfig(propertyName);
            return propertyName === 'description' ? (
              <textarea
                value={value}
                onChange={(e) => setValue(e.target.value)}
                className={`border-input focus-visible:ring-ring flex w-full rounded-md border bg-transparent px-3 py-2 text-sm shadow-sm transition-colors focus-visible:ring-1 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50 ${config.className}`}
                style={config.style}
                disabled={isSubmitting}
              />
            ) : (
              <textarea
                value={value}
                onChange={(e) => setValue(e.target.value)}
                rows={config.rows}
                className={`border-input focus-visible:ring-ring flex w-full rounded-md border bg-transparent px-3 py-2 text-sm shadow-sm transition-colors focus-visible:ring-1 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50 ${config.className}`}
                disabled={isSubmitting}
              />
            );
          })()}
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
