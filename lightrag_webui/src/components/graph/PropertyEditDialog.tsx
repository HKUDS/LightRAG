import Button from '@/components/ui/Button'
import Checkbox from '@/components/ui/Checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/Dialog'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

interface PropertyEditDialogProps {
  isOpen: boolean
  onClose: () => void
  onSave: (value: string, options?: { allowMerge?: boolean }) => void
  propertyName: string
  initialValue: string
  isSubmitting?: boolean
  errorMessage?: string | null
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
  isSubmitting = false,
  errorMessage = null,
}: PropertyEditDialogProps) => {
  const { t } = useTranslation()
  const [value, setValue] = useState('')
  const [allowMerge, setAllowMerge] = useState(false)

  // Initialize value when dialog opens
  useEffect(() => {
    if (isOpen) {
      setValue(initialValue)
      setAllowMerge(false)
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
            resize: 'vertical' as const, // Allow vertical resizing, using 'as const' to fix type
          },
        }
      case 'entity_id':
        return {
          rows: 2,
          className: '',
          style: {},
        }
      case 'keywords':
        return {
          rows: 4,
          className: '',
          style: {},
        }
      default:
        return {
          rows: 5,
          className: '',
          style: {},
        }
    }
  }

  const handleSave = async () => {
    const trimmedValue = value.trim()
    if (trimmedValue !== '') {
      const options = propertyName === 'entity_id' ? { allowMerge } : undefined
      await onSave(trimmedValue, options)
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {t('graphPanel.propertiesView.editProperty', {
              property: getPropertyNameTranslation(propertyName),
            })}
          </DialogTitle>
          <DialogDescription>
            {t('graphPanel.propertiesView.editPropertyDescription')}
          </DialogDescription>
        </DialogHeader>

        {/* Display error message if save fails */}
        {errorMessage && (
          <div className="bg-destructive/15 text-destructive px-4 py-2 rounded-md text-sm">
            {errorMessage}
          </div>
        )}

        {/* Multi-line text input using textarea */}
        <div className="grid gap-4 py-4">
          {(() => {
            const config = getTextareaConfig(propertyName)
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
            )
          })()}
        </div>

        {propertyName === 'entity_id' && (
          <div className="rounded-md border border-border bg-muted/20 p-3">
            <label className="flex items-start gap-2 text-sm font-medium">
              <Checkbox
                id="allow-merge"
                checked={allowMerge}
                disabled={isSubmitting}
                onCheckedChange={(checked) => setAllowMerge(checked === true)}
              />
              <div>
                <span>{t('graphPanel.propertiesView.mergeOptionLabel')}</span>
                <p className="text-xs font-normal text-muted-foreground">
                  {t('graphPanel.propertiesView.mergeOptionDescription')}
                </p>
              </div>
            </label>
          </div>
        )}

        <DialogFooter>
          <Button type="button" variant="outline" onClick={onClose} disabled={isSubmitting}>
            {t('common.cancel')}
          </Button>
          <Button type="button" onClick={handleSave} disabled={isSubmitting}>
            {isSubmitting ? (
              <>
                <span className="mr-2">
                  <svg
                    className="animate-spin h-4 w-4"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                </span>
                {t('common.saving')}
              </>
            ) : (
              t('common.save')
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default PropertyEditDialog
