import { QueryPromptOverrides } from '@/api/lightrag'
import Button from '@/components/ui/Button'
import Textarea from '@/components/ui/Textarea'
import { useTranslation } from 'react-i18next'
import { pruneEmptyPromptOverrides, setExampleItem } from '@/utils/promptOverrides'

type PromptOverridesEditorProps = {
  enabled: boolean
  disabledReason?: string
  value?: QueryPromptOverrides
  onChange: (value: QueryPromptOverrides | undefined) => void
}

export default function PromptOverridesEditor({ enabled, disabledReason, value, onChange }: PromptOverridesEditorProps) {
  const { t } = useTranslation()
  const examples = value?.keywords?.keywords_extraction_examples || []

  const updateValue = (next: QueryPromptOverrides) => {
    onChange(pruneEmptyPromptOverrides(next))
  }

  const setQueryField = (field: keyof NonNullable<QueryPromptOverrides['query']>, nextValue: string) => {
    updateValue({
      ...(value || {}),
      query: {
        ...value?.query,
        [field]: nextValue
      }
    })
  }

  const setKeywordsField = (
    field: keyof NonNullable<QueryPromptOverrides['keywords']>,
    nextValue: string | string[]
  ) => {
    updateValue({
      ...(value || {}),
      keywords: {
        ...value?.keywords,
        [field]: nextValue
      }
    })
  }

  const setExamples = (nextExamples: string[]) => {
    setKeywordsField('keywords_extraction_examples', nextExamples)
  }

  const shownExamples = [...examples, '']

  return (
    <div className="space-y-1.5 rounded-md border border-border/70 p-2">
      <label className="ml-1">{t('retrievePanel.querySettings.promptOverrides.title')}</label>
      <p className="ml-1 text-[11px] text-muted-foreground">
        {t('retrievePanel.querySettings.promptOverrides.description')}
      </p>
      {!enabled && (
        <p className="ml-1 text-[11px] text-amber-600 dark:text-amber-400">
          {disabledReason || t('retrievePanel.querySettings.promptOverrides.disabledHint')}
        </p>
      )}

      <div className="space-y-2">
        <div className="space-y-1">
          <label className="ml-1 text-[11px]">{t('retrievePanel.querySettings.promptOverrides.queryRagResponse')}</label>
          <Textarea
            value={value?.query?.rag_response || ''}
            onChange={(e) => setQueryField('rag_response', e.target.value)}
            placeholder="{context_data}"
            className="min-h-[70px] py-1.5 text-xs"
            disabled={!enabled}
          />
        </div>

        <div className="space-y-1">
          <label className="ml-1 text-[11px]">{t('retrievePanel.querySettings.promptOverrides.queryNaiveRagResponse')}</label>
          <Textarea
            value={value?.query?.naive_rag_response || ''}
            onChange={(e) => setQueryField('naive_rag_response', e.target.value)}
            placeholder="{content_data}"
            className="min-h-[70px] py-1.5 text-xs"
            disabled={!enabled}
          />
        </div>

        <div className="space-y-1">
          <label className="ml-1 text-[11px]">{t('retrievePanel.querySettings.promptOverrides.queryKgQueryContext')}</label>
          <Textarea
            value={value?.query?.kg_query_context || ''}
            onChange={(e) => setQueryField('kg_query_context', e.target.value)}
            placeholder="{entities_str}\n{relations_str}\n{text_chunks_str}\n{reference_list_str}"
            className="min-h-[70px] py-1.5 text-xs"
            disabled={!enabled}
          />
        </div>

        <div className="space-y-1">
          <label className="ml-1 text-[11px]">{t('retrievePanel.querySettings.promptOverrides.queryNaiveQueryContext')}</label>
          <Textarea
            value={value?.query?.naive_query_context || ''}
            onChange={(e) => setQueryField('naive_query_context', e.target.value)}
            placeholder="{text_chunks_str}\n{reference_list_str}"
            className="min-h-[70px] py-1.5 text-xs"
            disabled={!enabled}
          />
        </div>

        <div className="space-y-1">
          <label className="ml-1 text-[11px]">{t('retrievePanel.querySettings.promptOverrides.keywordsExtraction')}</label>
          <Textarea
            value={value?.keywords?.keywords_extraction || ''}
            onChange={(e) => setKeywordsField('keywords_extraction', e.target.value)}
            placeholder="{query}\n{examples}\n{language}"
            className="min-h-[70px] py-1.5 text-xs"
            disabled={!enabled}
          />
        </div>

        <div className="space-y-1">
          <label className="ml-1 text-[11px]">{t('retrievePanel.querySettings.promptOverrides.keywordsExtractionExamples')}</label>
          <div className="space-y-1">
            {shownExamples.map((example, index) => {
              const isPersistedItem = index < examples.length
              return (
                <div key={`keywords-example-${index}`} className="flex items-center gap-1">
                  <Textarea
                    value={example}
                    onChange={(e) => {
                      const nextValue = e.target.value
                      if (isPersistedItem) {
                        setExamples(setExampleItem(examples, index, nextValue))
                        return
                      }
                      if (nextValue.trim().length > 0) {
                        setExamples([...examples, nextValue])
                      }
                    }}
                    placeholder={t('retrievePanel.querySettings.promptOverrides.examplePlaceholder', {
                      index: index + 1
                    })}
                    className="min-h-[52px] py-1.5 text-xs"
                    disabled={!enabled}
                  />
                  {isPersistedItem && (
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-8 px-2 text-[11px]"
                      disabled={!enabled}
                      onClick={() => setExamples(examples.filter((_, itemIndex) => itemIndex !== index))}
                    >
                      {t('retrievePanel.querySettings.promptOverrides.removeExample')}
                    </Button>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
