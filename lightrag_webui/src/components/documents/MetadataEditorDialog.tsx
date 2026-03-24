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
import { toast } from 'sonner'
import { errorMessage } from '@/lib/utils'
import { updateDocumentMetadata } from '@/api/lightrag'

import { PencilIcon, PlusIcon, XIcon, LockIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

// System-reserved metadata fields that cannot be modified
const RESERVED_METADATA_FIELDS = new Set([
  'is_duplicate',
  'original_doc_id',
  'original_track_id',
  'error_type',
  'processing_start_time',
  'processing_end_time'
])

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

interface MetadataRow {
  id: string
  key: string
  value: string
  isReserved: boolean
}

interface MetadataEditorDialogProps {
  documentId: string
  documentName?: string
  currentMetadata?: Record<string, any>
  onMetadataUpdated?: () => Promise<void>
  triggerButton?: React.ReactNode
}

export default function MetadataEditorDialog({
  documentId,
  documentName,
  currentMetadata = {},
  onMetadataUpdated,
  triggerButton
}: MetadataEditorDialogProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [rows, setRows] = useState<MetadataRow[]>([])
  const [isSaving, setIsSaving] = useState(false)

  // Initialize rows from current metadata when dialog opens
  useEffect(() => {
    if (open) {
      // Separate reserved and user fields, keeping both in original order
      const entries = Object.entries(currentMetadata)
      const reservedEntries = entries.filter(([key]) => RESERVED_METADATA_FIELDS.has(key))
      const userEntries = entries.filter(([key]) => !RESERVED_METADATA_FIELDS.has(key))

      // Reserved fields first, then user fields (both in original order)
      const orderedEntries = [...reservedEntries, ...userEntries]

      const initialRows: MetadataRow[] = orderedEntries.map(
        ([key, value], index) => ({
          id: `initial-${index}`,
          key,
          value: typeof value === 'object' ? JSON.stringify(value) : String(value),
          isReserved: RESERVED_METADATA_FIELDS.has(key)
        })
      )
      // Always start with at least one empty row if no metadata exists
      if (initialRows.length === 0) {
        initialRows.push({ id: 'new-0', key: '', value: '', isReserved: false })
      }
      setRows(initialRows)
    }
  }, [open, currentMetadata])

  // Add a new empty row
  const handleAddRow = useCallback(() => {
    const newRow: MetadataRow = {
      id: `new-${Date.now()}`,
      key: '',
      value: '',
      isReserved: false
    }
    setRows((prev) => [...prev, newRow])
  }, [])

  // Delete a row (only for non-reserved fields)
  const handleDeleteRow = useCallback((rowId: string) => {
    setRows((prev) => {
      const newRows = prev.filter((row) => row.id !== rowId)
      // Ensure at least one row remains
      if (newRows.length === 0) {
        return [{ id: `new-${Date.now()}`, key: '', value: '', isReserved: false }]
      }
      return newRows
    })
  }, [])

  // Update a row's key or value (only for non-reserved fields)
  const handleUpdateRow = useCallback((rowId: string, field: 'key' | 'value', newValue: string) => {
    setRows((prev) =>
      prev.map((row) =>
        row.id === rowId ? { ...row, [field]: newValue } : row
      )
    )
  }, [])

  // Save metadata
  const handleSave = useCallback(async () => {
    // Validate: no duplicate keys, no empty keys for non-empty values
    const metadata: Record<string, any> = {}
    const keys = new Set<string>()

    for (const row of rows) {
      // Skip reserved fields - they are read-only
      if (row.isReserved) {
        continue
      }

      const trimmedKey = row.key.trim()
      const trimmedValue = row.value.trim()

      // Skip completely empty rows
      if (!trimmedKey && !trimmedValue) {
        continue
      }

      // Check for empty key with non-empty value
      if (!trimmedKey && trimmedValue) {
        toast.error(t('documentPanel.metadataEditor.errorEmptyKey'))
        return
      }

      // Check for duplicate keys
      if (keys.has(trimmedKey)) {
        toast.error(t('documentPanel.metadataEditor.errorDuplicateKey', { key: trimmedKey }))
        return
      }

      // Check if user is trying to use a reserved field name
      if (RESERVED_METADATA_FIELDS.has(trimmedKey)) {
        toast.error(t('documentPanel.metadataEditor.errorReservedKey', { key: trimmedKey }))
        return
      }

      keys.add(trimmedKey)

      // Try to parse value as JSON, otherwise use as string
      try {
        metadata[trimmedKey] = JSON.parse(trimmedValue)
      } catch {
        metadata[trimmedKey] = trimmedValue
      }
    }

    setIsSaving(true)
    try {
      await updateDocumentMetadata(documentId, metadata)
      toast.success(t('documentPanel.metadataEditor.success'))

      // Refresh document list if provided
      if (onMetadataUpdated) {
        await onMetadataUpdated()
      }

      // Close dialog
      setOpen(false)
    } catch (err) {
      toast.error(t('documentPanel.metadataEditor.error', { error: errorMessage(err) }))
    } finally {
      setIsSaving(false)
    }
  }, [rows, documentId, t, onMetadataUpdated])

  const defaultTrigger = (
    <Button
      variant="ghost"
      size="sm"
      tooltip={t('documentPanel.metadataEditor.tooltip')}
    >
      <PencilIcon className="h-4 w-4" />
    </Button>
  )

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        {triggerButton || defaultTrigger}
      </DialogTrigger>
      <DialogContent className="sm:max-w-3xl max-h-[80vh] flex flex-col" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <PencilIcon className="h-5 w-5" />
            {t('documentPanel.metadataEditor.title')}
          </DialogTitle>
          <DialogDescription>
            {documentName
              ? t('documentPanel.metadataEditor.descriptionWithName', { name: documentName })
              : t('documentPanel.metadataEditor.description', { id: documentId })}
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto space-y-2 py-4 px-1">
          <div className="grid grid-cols-[1fr_1fr_auto] gap-2 mb-2 font-semibold text-sm">
            <Label>{t('documentPanel.metadataEditor.keyLabel')}</Label>
            <Label>{t('documentPanel.metadataEditor.valueLabel')}</Label>
            <div className="w-8"></div>
          </div>

          {rows.map((row) => (
            <div
              key={row.id}
              className={`grid grid-cols-[1fr_1fr_auto] gap-2 items-center ${
                row.isReserved ? 'opacity-60' : ''
              }`}
            >
              <div className="relative">
                <Input
                  value={row.key}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    handleUpdateRow(row.id, 'key', e.target.value)
                  }
                  placeholder={t('documentPanel.metadataEditor.keyPlaceholder')}
                  disabled={isSaving || row.isReserved}
                  className={`font-mono text-sm ${row.isReserved ? 'bg-muted cursor-not-allowed' : ''}`}
                />
                {row.isReserved && (
                  <LockIcon className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                )}
              </div>
              <Input
                value={row.value}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                  handleUpdateRow(row.id, 'value', e.target.value)
                }
                placeholder={t('documentPanel.metadataEditor.valuePlaceholder')}
                disabled={isSaving || row.isReserved}
                className={`font-mono text-sm ${row.isReserved ? 'bg-muted cursor-not-allowed' : ''}`}
              />
              {row.isReserved ? (
                <div
                  className="h-8 w-8 flex items-center justify-center"
                  title={t('documentPanel.metadataEditor.systemField')}
                >
                  <LockIcon className="h-4 w-4 text-muted-foreground" />
                </div>
              ) : (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleDeleteRow(row.id)}
                  disabled={isSaving}
                  tooltip={t('documentPanel.metadataEditor.deleteRowTooltip')}
                  className="h-8 w-8 p-0"
                >
                  <XIcon className="h-4 w-4" />
                </Button>
              )}
            </div>
          ))}

          <Button
            variant="outline"
            size="sm"
            onClick={handleAddRow}
            disabled={isSaving}
            className="w-full mt-4"
          >
            <PlusIcon className="h-4 w-4 mr-2" />
            {t('documentPanel.metadataEditor.addRowButton')}
          </Button>
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => setOpen(false)}
            disabled={isSaving}
          >
            {t('documentPanel.metadataEditor.cancelButton')}
          </Button>
          <Button
            onClick={handleSave}
            disabled={isSaving}
          >
            {isSaving ? t('documentPanel.metadataEditor.savingButton') : t('documentPanel.metadataEditor.saveButton')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
