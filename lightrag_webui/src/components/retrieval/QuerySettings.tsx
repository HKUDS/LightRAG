import { useCallback } from 'react'
import { QueryMode, QueryRequest } from '@/api/lightrag'
import Text from '@/components/ui/Text'
import Input from '@/components/ui/Input'
import Checkbox from '@/components/ui/Checkbox'
import NumberInput from '@/components/ui/NumberInput'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import { useSettingsStore } from '@/stores/settings'
import { useTranslation } from 'react-i18next'

export default function QuerySettings() {
  const { t } = useTranslation()
  const querySettings = useSettingsStore((state) => state.querySettings)

  const handleChange = useCallback((key: keyof QueryRequest, value: any) => {
    useSettingsStore.getState().updateQuerySettings({ [key]: value })
  }, [])

  return (
    <Card className="flex shrink-0 flex-col min-w-[220px]">
      <CardHeader className="px-4 pt-4 pb-2">
        <CardTitle>{t('retrievePanel.querySettings.parametersTitle')}</CardTitle>
        <CardDescription>{t('retrievePanel.querySettings.parametersDescription')}</CardDescription>
      </CardHeader>
      <CardContent className="m-0 flex grow flex-col p-0 text-xs">
        <div className="relative size-full">
          <div className="absolute inset-0 flex flex-col gap-2 overflow-auto px-2">
            {/* Query Mode */}
            <>
              <Text
                className="ml-1"
                text={t('retrievePanel.querySettings.queryMode')}
                tooltip={t('retrievePanel.querySettings.queryModeTooltip')}
                side="left"
              />
              <Select
                value={querySettings.mode}
                onValueChange={(v) => handleChange('mode', v as QueryMode)}
              >
                <SelectTrigger className="hover:bg-primary/5 h-9 cursor-pointer focus:ring-0 focus:ring-offset-0 focus:outline-0 active:right-0">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectItem value="naive">{t('retrievePanel.querySettings.queryModeOptions.naive')}</SelectItem>
                    <SelectItem value="local">{t('retrievePanel.querySettings.queryModeOptions.local')}</SelectItem>
                    <SelectItem value="global">{t('retrievePanel.querySettings.queryModeOptions.global')}</SelectItem>
                    <SelectItem value="hybrid">{t('retrievePanel.querySettings.queryModeOptions.hybrid')}</SelectItem>
                    <SelectItem value="mix">{t('retrievePanel.querySettings.queryModeOptions.mix')}</SelectItem>
                    <SelectItem value="bypass">{t('retrievePanel.querySettings.queryModeOptions.bypass')}</SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
            </>

            {/* Response Format */}
            <>
              <Text
                className="ml-1"
                text={t('retrievePanel.querySettings.responseFormat')}
                tooltip={t('retrievePanel.querySettings.responseFormatTooltip')}
                side="left"
              />
              <Select
                value={querySettings.response_type}
                onValueChange={(v) => handleChange('response_type', v)}
              >
                <SelectTrigger className="hover:bg-primary/5 h-9 cursor-pointer focus:ring-0 focus:ring-offset-0 focus:outline-0 active:right-0">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectItem value="Multiple Paragraphs">{t('retrievePanel.querySettings.responseFormatOptions.multipleParagraphs')}</SelectItem>
                    <SelectItem value="Single Paragraph">{t('retrievePanel.querySettings.responseFormatOptions.singleParagraph')}</SelectItem>
                    <SelectItem value="Bullet Points">{t('retrievePanel.querySettings.responseFormatOptions.bulletPoints')}</SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
            </>

            {/* Top K */}
            <>
              <Text
                className="ml-1"
                text={t('retrievePanel.querySettings.topK')}
                tooltip={t('retrievePanel.querySettings.topKTooltip')}
                side="left"
              />
              <div>
                <label htmlFor="top_k" className="sr-only">
                  {t('retrievePanel.querySettings.topK')}
                </label>
                <NumberInput
                  id="top_k"
                  stepper={1}
                  value={querySettings.top_k}
                  onValueChange={(v) => handleChange('top_k', v)}
                  min={1}
                  placeholder={t('retrievePanel.querySettings.topKPlaceholder')}
                />
              </div>
            </>

            {/* Max Tokens */}
            <>
              <>
                <Text
                  className="ml-1"
                  text={t('retrievePanel.querySettings.maxTokensTextUnit')}
                  tooltip={t('retrievePanel.querySettings.maxTokensTextUnitTooltip')}
                  side="left"
                />
                <div>
                  <label htmlFor="max_token_for_text_unit" className="sr-only">
                    {t('retrievePanel.querySettings.maxTokensTextUnit')}
                  </label>
                  <NumberInput
                    id="max_token_for_text_unit"
                    stepper={500}
                    value={querySettings.max_token_for_text_unit}
                    onValueChange={(v) => handleChange('max_token_for_text_unit', v)}
                    min={1}
                    placeholder={t('retrievePanel.querySettings.maxTokensTextUnit')}
                  />
                </div>
              </>

              <>
                <Text
                  text={t('retrievePanel.querySettings.maxTokensGlobalContext')}
                  tooltip={t('retrievePanel.querySettings.maxTokensGlobalContextTooltip')}
                  side="left"
                />
                <div>
                  <label htmlFor="max_token_for_global_context" className="sr-only">
                    {t('retrievePanel.querySettings.maxTokensGlobalContext')}
                  </label>
                  <NumberInput
                    id="max_token_for_global_context"
                    stepper={500}
                    value={querySettings.max_token_for_global_context}
                    onValueChange={(v) => handleChange('max_token_for_global_context', v)}
                    min={1}
                    placeholder={t('retrievePanel.querySettings.maxTokensGlobalContext')}
                  />
                </div>
              </>

              <>
                <Text
                  className="ml-1"
                  text={t('retrievePanel.querySettings.maxTokensLocalContext')}
                  tooltip={t('retrievePanel.querySettings.maxTokensLocalContextTooltip')}
                  side="left"
                />
                <div>
                  <label htmlFor="max_token_for_local_context" className="sr-only">
                    {t('retrievePanel.querySettings.maxTokensLocalContext')}
                  </label>
                  <NumberInput
                    id="max_token_for_local_context"
                    stepper={500}
                    value={querySettings.max_token_for_local_context}
                    onValueChange={(v) => handleChange('max_token_for_local_context', v)}
                    min={1}
                    placeholder={t('retrievePanel.querySettings.maxTokensLocalContext')}
                  />
                </div>
              </>
            </>

            {/* History Turns */}
            <>
              <Text
                className="ml-1"
                text={t('retrievePanel.querySettings.historyTurns')}
                tooltip={t('retrievePanel.querySettings.historyTurnsTooltip')}
                side="left"
              />
              <div>
                <label htmlFor="history_turns" className="sr-only">
                  {t('retrievePanel.querySettings.historyTurns')}
                </label>
                <NumberInput
                  className="!border-input"
                  id="history_turns"
                  stepper={1}
                  type="text"
                  value={querySettings.history_turns}
                  onValueChange={(v) => handleChange('history_turns', v)}
                  min={0}
                  placeholder={t('retrievePanel.querySettings.historyTurnsPlaceholder')}
                />
              </div>
            </>

            {/* Keywords */}
            <>
              <>
                <Text
                  className="ml-1"
                  text={t('retrievePanel.querySettings.hlKeywords')}
                  tooltip={t('retrievePanel.querySettings.hlKeywordsTooltip')}
                  side="left"
                />
                <div>
                  <label htmlFor="hl_keywords" className="sr-only">
                    {t('retrievePanel.querySettings.hlKeywords')}
                  </label>
                  <Input
                    id="hl_keywords"
                    type="text"
                    value={querySettings.hl_keywords?.join(', ')}
                    onChange={(e) => {
                      const keywords = e.target.value
                        .split(',')
                        .map((k) => k.trim())
                        .filter((k) => k !== '')
                      handleChange('hl_keywords', keywords)
                    }}
                    placeholder={t('retrievePanel.querySettings.hlkeywordsPlaceHolder')}
                  />
                </div>
              </>

              <>
                <Text
                  className="ml-1"
                  text={t('retrievePanel.querySettings.llKeywords')}
                  tooltip={t('retrievePanel.querySettings.llKeywordsTooltip')}
                  side="left"
                />
                <div>
                  <label htmlFor="ll_keywords" className="sr-only">
                    {t('retrievePanel.querySettings.llKeywords')}
                  </label>
                  <Input
                    id="ll_keywords"
                    type="text"
                    value={querySettings.ll_keywords?.join(', ')}
                    onChange={(e) => {
                      const keywords = e.target.value
                        .split(',')
                        .map((k) => k.trim())
                        .filter((k) => k !== '')
                      handleChange('ll_keywords', keywords)
                    }}
                    placeholder={t('retrievePanel.querySettings.hlkeywordsPlaceHolder')}
                  />
                </div>
              </>
            </>

            {/* Toggle Options */}
            <>
              <div className="flex items-center gap-2">
                <label htmlFor="only_need_context" className="flex-1">
                  <Text
                    className="ml-1"
                    text={t('retrievePanel.querySettings.onlyNeedContext')}
                    tooltip={t('retrievePanel.querySettings.onlyNeedContextTooltip')}
                    side="left"
                  />
                </label>
                <Checkbox
                  className="mr-1 cursor-pointer"
                  id="only_need_context"
                  checked={querySettings.only_need_context}
                  onCheckedChange={(checked) => handleChange('only_need_context', checked)}
                />
              </div>

              <div className="flex items-center gap-2">
                <label htmlFor="only_need_prompt" className="flex-1">
                  <Text
                    className="ml-1"
                    text={t('retrievePanel.querySettings.onlyNeedPrompt')}
                    tooltip={t('retrievePanel.querySettings.onlyNeedPromptTooltip')}
                    side="left"
                  />
                </label>
                <Checkbox
                  className="mr-1 cursor-pointer"
                  id="only_need_prompt"
                  checked={querySettings.only_need_prompt}
                  onCheckedChange={(checked) => handleChange('only_need_prompt', checked)}
                />
              </div>

              <div className="flex items-center gap-2">
                <label htmlFor="stream" className="flex-1">
                  <Text
                    className="ml-1"
                    text={t('retrievePanel.querySettings.streamResponse')}
                    tooltip={t('retrievePanel.querySettings.streamResponseTooltip')}
                    side="left"
                  />
                </label>
                <Checkbox
                  className="mr-1 cursor-pointer"
                  id="stream"
                  checked={querySettings.stream}
                  onCheckedChange={(checked) => handleChange('stream', checked)}
                />
              </div>
            </>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
