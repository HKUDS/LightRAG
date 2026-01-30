import { useState } from 'react'
import { ChevronDownIcon, ChevronUpIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

interface EntityTypeConfigProps {
  entityTypes: string[]
  onEntityTypesChange: (types: string[]) => void
  defaultEntityTypes: string[]
}

export default function EntityTypeConfig({
  entityTypes,
  onEntityTypesChange,
  defaultEntityTypes
}: EntityTypeConfigProps) {
  const { t } = useTranslation()
  const [isExpanded, setIsExpanded] = useState(false)
  const [useDefault, setUseDefault] = useState(true)
  const [textInput, setTextInput] = useState('')

  const handleUseDefaultToggle = (checked: boolean) => {
    setUseDefault(checked)
    if (checked) {
      onEntityTypesChange([])
      setTextInput('')
    }
  }

  const handleTextInputChange = (value: string) => {
    setTextInput(value)
    const lines = value
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0)
    onEntityTypesChange(lines)
    // 只在用户明确点击复选框时才使用默认配置，不在输入为空时自动切换
    // 这样用户可以清空输入框重新输入，而不会导致输入框消失
  }

  return (
    <div className="space-y-2">
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 text-sm font-medium text-foreground/80 hover:text-foreground transition-colors"
      >
        {isExpanded ? (
          <ChevronUpIcon className="w-4 h-4" />
        ) : (
          <ChevronDownIcon className="w-4 h-4" />
        )}
        {t('documentPanel.entityTypeConfig.advancedSettings')}
      </button>

      {isExpanded && (
        <div className="pl-6 space-y-3">
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="use-default-entity-types"
              checked={useDefault}
              onChange={e => handleUseDefaultToggle(e.target.checked)}
              className="w-4 h-4 rounded border-gray-300 text-primary focus:ring-primary"
            />
            <label
              htmlFor="use-default-entity-types"
              className="text-sm text-foreground/70"
            >
              {t('documentPanel.entityTypeConfig.useDefaultForUpload')}
            </label>
          </div>

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
                    count: entityTypes.length
                  })}
                </span>
                <span>{t('documentPanel.entityTypeConfig.hint')}</span>
              </div>

              {defaultEntityTypes.length > 0 && (
                <div className="mt-2 p-2 bg-muted/50 rounded text-xs">
                  <div className="font-medium text-foreground/70 mb-1">
                    {t('documentPanel.entityTypeConfig.defaultLabel')}
                  </div>
                  <div className="text-foreground/60">
                    {defaultEntityTypes.join(', ')}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
