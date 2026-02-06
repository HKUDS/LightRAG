import { useState } from 'react'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'
import { useTranslation } from 'react-i18next'

interface ScanConfigDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onConfirm: (entityTypes: string[] | undefined) => void
  defaultEntityTypes: string[]
  isScanning?: boolean
}

export default function ScanConfigDialog({
  open,
  onOpenChange,
  onConfirm,
  defaultEntityTypes,
  isScanning = false
}: ScanConfigDialogProps) {
  const { t } = useTranslation()
  const [useDefault, setUseDefault] = useState<boolean>(true)
  const [customEntityTypes, setCustomEntityTypes] = useState<string[]>([])
  const [textInput, setTextInput] = useState<string>('')

  const handleConfirm = () => {
    // 如果使用默认类型，传递 undefined
    // 否则传递自定义的类型
    const entityTypesToUse = useDefault ? undefined : customEntityTypes
    onConfirm(entityTypesToUse)
    // 关闭弹窗
    onOpenChange(false)
  }

  const handleCancel = () => {
    onOpenChange(false)
    // 重置状态
    setUseDefault(true)
    setCustomEntityTypes([])
    setTextInput('')
  }

  // 当对话框关闭时重置状态
  const handleOpenChange = (newOpen: boolean) => {
    if (!newOpen) {
      setUseDefault(true)
      setCustomEntityTypes([])
      setTextInput('')
    }
    onOpenChange(newOpen)
  }

  const handleCheckboxChange = (checked: boolean) => {
    setUseDefault(checked)
    if (checked) {
      setTextInput('')
      setCustomEntityTypes([])
    }
  }

  const handleTextInputChange = (value: string) => {
    setTextInput(value)
    const lines = value
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0)
    setCustomEntityTypes(lines)
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-lg overflow-hidden">
        <DialogHeader>
          <DialogTitle>{t('documentPanel.documentManager.scanButton')}</DialogTitle>
          <DialogDescription>
            {t('documentPanel.documentManager.scanConfig.description')}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="use-default-entity-types-scan"
                checked={useDefault}
                onChange={e => handleCheckboxChange(e.target.checked)}
                className="w-4 h-4 rounded border-gray-300 text-primary focus:ring-primary"
              />
              <label
                htmlFor="use-default-entity-types-scan"
                className="text-sm text-foreground/70"
              >
                {t('documentPanel.entityTypeConfig.useDefaultForScan')}
              </label>
            </div>

            {useDefault && defaultEntityTypes.length > 0 && (
              <div className="pl-6 p-2 bg-muted/50 rounded text-xs">
                <div className="font-medium text-foreground/70 mb-1">
                  {t('documentPanel.entityTypeConfig.defaultLabel')}
                </div>
                <div className="text-foreground/60">
                  {defaultEntityTypes.join(', ')}
                </div>
              </div>
            )}

            {!useDefault && (
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground/80">
                  {t('documentPanel.entityTypeConfig.customTypes')}
                </label>
                <textarea
                  value={textInput}
                  onChange={e => handleTextInputChange(e.target.value)}
                  placeholder={t('documentPanel.entityTypeConfig.placeholder')}
                  rows={6}
                  className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent resize-y"
                />
                <div className="flex items-center justify-between text-xs text-foreground/60">
                  <span>
                    {t('documentPanel.entityTypeConfig.count', {
                      count: customEntityTypes.length
                    })}
                  </span>
                  <span>{t('documentPanel.entityTypeConfig.hint')}</span>
                </div>
              </div>
            )}
          </div>
        </div>

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={handleCancel}
            disabled={isScanning}
          >
            {t('documentPanel.documentManager.scanConfig.cancel')}
          </Button>
          <Button
            type="button"
            variant="default"
            onClick={handleConfirm}
            disabled={isScanning}
          >
            {isScanning ? t('documentPanel.documentManager.scanConfig.scanning') : '启动'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
