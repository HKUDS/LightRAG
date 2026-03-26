import Button from '@/components/ui/Button'
import Textarea from '@/components/ui/Textarea'
import { useTranslation } from 'react-i18next'

type PromptListFieldEditorProps = {
  value: string[]
  onChange: (value: string[]) => void
  placeholder: string
  itemLabel: string
}

export default function PromptListFieldEditor({
  value,
  onChange,
  placeholder,
  itemLabel
}: PromptListFieldEditorProps) {
  const { t } = useTranslation()
  const items = [...value, '']

  const pruneItems = (itemsToPrune: string[]) => itemsToPrune.filter((entry) => entry.trim().length > 0)

  return (
    <div className="space-y-3">
      {items.map((item, index) => {
        const isPersisted = index < value.length
        return (
          <div key={`list-item-${index}`} className="rounded-lg border border-border/60 bg-muted/20 p-3">
            <div className="mb-2 flex items-center justify-between gap-2">
              <div className="text-xs font-medium text-muted-foreground">
                {itemLabel} {index + 1}
              </div>
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
            <Textarea
              value={item}
              onChange={(event) => {
                const nextValue = event.target.value
                if (isPersisted) {
                  const nextItems = [...value]
                  nextItems[index] = nextValue
                  onChange(pruneItems(nextItems))
                  return
                }
                if (nextValue.trim()) {
                  onChange([...value, nextValue])
                }
              }}
              placeholder={`${placeholder} ${index + 1}`}
              className="min-h-[160px] resize-y text-xs leading-5"
            />
          </div>
        )
      })}
    </div>
  )
}
