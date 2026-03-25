import Button from '@/components/ui/Button'
import Textarea from '@/components/ui/Textarea'
import { useTranslation } from 'react-i18next'

type PromptListFieldEditorProps = {
  value: string[]
  onChange: (value: string[]) => void
  placeholder: string
}

export default function PromptListFieldEditor({
  value,
  onChange,
  placeholder
}: PromptListFieldEditorProps) {
  const { t } = useTranslation()
  const items = [...value, '']

  return (
    <div className="space-y-2">
      {items.map((item, index) => {
        const isPersisted = index < value.length
        return (
          <div key={`list-item-${index}`} className="flex items-start gap-2">
            <Textarea
              value={item}
              onChange={(event) => {
                const nextValue = event.target.value
                if (isPersisted) {
                  const nextItems = [...value]
                  nextItems[index] = nextValue
                  onChange(nextItems.map((entry) => entry.trim()).filter(Boolean))
                  return
                }
                if (nextValue.trim()) {
                  onChange([...value, nextValue.trim()])
                }
              }}
              placeholder={`${placeholder} ${index + 1}`}
              className="min-h-[56px] text-xs"
            />
            {isPersisted ? (
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => onChange(value.filter((_, itemIndex) => itemIndex !== index))}
              >
                {t('promptManagement.remove')}
              </Button>
            ) : null}
          </div>
        )
      })}
    </div>
  )
}
