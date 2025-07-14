import { useCallback } from 'react'
import { QueryMode, QueryRequest } from '@/api/lightrag'
// Removed unused import for Text component
import Checkbox from '@/components/ui/Checkbox'
import NumberInput from '@/components/ui/NumberInput'
import Input from '@/components/ui/Input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'
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
        <CardDescription className="sr-only">{t('retrievePanel.querySettings.parametersDescription')}</CardDescription>
      </CardHeader>
      <CardContent className="m-0 flex grow flex-col p-0 text-xs">
        <div className="relative size-full">
          <div className="absolute inset-0 flex flex-col gap-2 overflow-auto px-2">
            {/* Query Mode */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="query_mode_select" className="ml-1 cursor-help">
                      {t('retrievePanel.querySettings.queryMode')}
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>{t('retrievePanel.querySettings.queryModeTooltip')}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <Select
                value={querySettings.mode}
                onValueChange={(v) => handleChange('mode', v as QueryMode)}
              >
                <SelectTrigger
                  id="query_mode_select"
                  className="hover:bg-primary/5 h-9 cursor-pointer focus:ring-0 focus:ring-offset-0 focus:outline-0 active:right-0"
                >
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
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="response_format_select" className="ml-1 cursor-help">
                      {t('retrievePanel.querySettings.responseFormat')}
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>{t('retrievePanel.querySettings.responseFormatTooltip')}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <Select
                value={querySettings.response_type}
                onValueChange={(v) => handleChange('response_type', v)}
              >
                <SelectTrigger
                  id="response_format_select"
                  className="hover:bg-primary/5 h-9 cursor-pointer focus:ring-0 focus:ring-offset-0 focus:outline-0 active:right-0"
                >
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
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="top_k" className="ml-1 cursor-help">
                      {t('retrievePanel.querySettings.topK')}
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>{t('retrievePanel.querySettings.topKTooltip')}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div>
                {/* Removed sr-only label */}
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

            {/* Chunk Top K */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="chunk_top_k" className="ml-1 cursor-help">
                      {t('retrievePanel.querySettings.chunkTopK')}
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>{t('retrievePanel.querySettings.chunkTopKTooltip')}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div>
                <NumberInput
                  id="chunk_top_k"
                  stepper={1}
                  value={querySettings.chunk_top_k}
                  onValueChange={(v) => handleChange('chunk_top_k', v)}
                  min={1}
                  placeholder={t('retrievePanel.querySettings.chunkTopKPlaceholder')}
                />
              </div>
            </>

            {/* Chunk Rerank Top K */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="chunk_rerank_top_k" className="ml-1 cursor-help">
                      {t('retrievePanel.querySettings.chunkRerankTopK')}
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>{t('retrievePanel.querySettings.chunkRerankTopKTooltip')}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div>
                <NumberInput
                  id="chunk_rerank_top_k"
                  stepper={1}
                  value={querySettings.chunk_rerank_top_k}
                  onValueChange={(v) => handleChange('chunk_rerank_top_k', v)}
                  min={1}
                  placeholder={t('retrievePanel.querySettings.chunkRerankTopKPlaceholder')}
                />
              </div>
            </>

            {/* Max Tokens */}
            <>
              <>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="max_entity_tokens" className="ml-1 cursor-help">
                        {t('retrievePanel.querySettings.maxEntityTokens')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.maxEntityTokensTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div>
                  <NumberInput
                    id="max_entity_tokens"
                    stepper={500}
                    value={querySettings.max_entity_tokens}
                    onValueChange={(v) => handleChange('max_entity_tokens', v)}
                    min={1}
                    placeholder={t('retrievePanel.querySettings.maxEntityTokens')}
                  />
                </div>
              </>

              <>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="max_relation_tokens" className="ml-1 cursor-help">
                        {t('retrievePanel.querySettings.maxRelationTokens')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.maxRelationTokensTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div>
                  <NumberInput
                    id="max_relation_tokens"
                    stepper={500}
                    value={querySettings.max_relation_tokens}
                    onValueChange={(v) => handleChange('max_relation_tokens', v)}
                    min={1}
                    placeholder={t('retrievePanel.querySettings.maxRelationTokens')}
                  />
                </div>
              </>

              <>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="max_total_tokens" className="ml-1 cursor-help">
                        {t('retrievePanel.querySettings.maxTotalTokens')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.maxTotalTokensTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div>
                  <NumberInput
                    id="max_total_tokens"
                    stepper={1000}
                    value={querySettings.max_total_tokens}
                    onValueChange={(v) => handleChange('max_total_tokens', v)}
                    min={1}
                    placeholder={t('retrievePanel.querySettings.maxTotalTokens')}
                  />
                </div>
              </>
            </>

            {/* History Turns */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="history_turns" className="ml-1 cursor-help">
                      {t('retrievePanel.querySettings.historyTurns')}
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>{t('retrievePanel.querySettings.historyTurnsTooltip')}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div>
                {/* Removed sr-only label */}
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

            {/* User Prompt */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="user_prompt" className="ml-1 cursor-help">
                      {t('retrievePanel.querySettings.userPrompt')}
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>{t('retrievePanel.querySettings.userPromptTooltip')}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div>
                <Input
                  id="user_prompt"
                  value={querySettings.user_prompt}
                  onChange={(e) => handleChange('user_prompt', e.target.value)}
                  placeholder={t('retrievePanel.querySettings.userPromptPlaceholder')}
                  className="h-9"
                />
              </div>
            </>

            {/* Toggle Options */}
            <>
              <div className="flex items-center gap-2">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="only_need_context" className="flex-1 ml-1 cursor-help">
                        {t('retrievePanel.querySettings.onlyNeedContext')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.onlyNeedContextTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <Checkbox
                  className="mr-1 cursor-pointer"
                  id="only_need_context"
                  checked={querySettings.only_need_context}
                  onCheckedChange={(checked) => handleChange('only_need_context', checked)}
                />
              </div>

              <div className="flex items-center gap-2">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="only_need_prompt" className="flex-1 ml-1 cursor-help">
                        {t('retrievePanel.querySettings.onlyNeedPrompt')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.onlyNeedPromptTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <Checkbox
                  className="mr-1 cursor-pointer"
                  id="only_need_prompt"
                  checked={querySettings.only_need_prompt}
                  onCheckedChange={(checked) => handleChange('only_need_prompt', checked)}
                />
              </div>

              <div className="flex items-center gap-2">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="stream" className="flex-1 ml-1 cursor-help">
                        {t('retrievePanel.querySettings.streamResponse')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.streamResponseTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
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
