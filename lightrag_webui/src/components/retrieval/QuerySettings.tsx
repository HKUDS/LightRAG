import type { QueryMode, QueryRequest } from '@/api/lightrag'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
// Removed unused import for Text component
import Checkbox from '@/components/ui/Checkbox'
import Input from '@/components/ui/Input'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/Select'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'
import UserPromptInputWithHistory from '@/components/ui/UserPromptInputWithHistory'
import { useSettingsStore } from '@/stores/settings'
import { RotateCcw } from 'lucide-react'
import { useCallback, useMemo } from 'react'
import { useTranslation } from 'react-i18next'

export default function QuerySettings() {
  const { t } = useTranslation()
  const querySettings = useSettingsStore((state) => state.querySettings)
  const userPromptHistory = useSettingsStore((state) => state.userPromptHistory)

  const handleChange = useCallback((key: keyof QueryRequest, value: any) => {
    useSettingsStore.getState().updateQuerySettings({ [key]: value })
  }, [])

  const handleSelectFromHistory = useCallback(
    (prompt: string) => {
      handleChange('user_prompt', prompt)
    },
    [handleChange]
  )

  const handleDeleteFromHistory = useCallback(
    (index: number) => {
      const newHistory = [...userPromptHistory]
      newHistory.splice(index, 1)
      useSettingsStore.getState().setUserPromptHistory(newHistory)
    },
    [userPromptHistory]
  )

  // Default values for reset functionality
  const defaultValues = useMemo(
    () => ({
      mode: 'mix' as QueryMode,
      top_k: 40,
      chunk_top_k: 20,
      max_entity_tokens: 6000,
      max_relation_tokens: 8000,
      max_total_tokens: 30000,
      citation_mode: 'none' as 'none' | 'inline' | 'footnotes',
      citation_threshold: 0.7,
    }),
    []
  )

  const handleReset = useCallback(
    (key: keyof typeof defaultValues) => {
      handleChange(key, defaultValues[key])
    },
    [handleChange, defaultValues]
  )

  // Reset button component
  const ResetButton = ({ onClick, title }: { onClick: () => void; title: string }) => (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={onClick}
            className="mr-1 p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            title={title}
          >
            <RotateCcw className="h-3 w-3 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200" />
          </button>
        </TooltipTrigger>
        <TooltipContent side="left">
          <p>{title}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )

  return (
    <Card className="flex shrink-0 flex-col w-[280px]">
      <CardHeader className="px-4 pt-4 pb-2">
        <CardTitle>{t('retrievePanel.querySettings.parametersTitle')}</CardTitle>
        <CardDescription className="sr-only">
          {t('retrievePanel.querySettings.parametersDescription')}
        </CardDescription>
      </CardHeader>
      <CardContent className="m-0 flex grow flex-col p-0 text-xs">
        <div className="relative size-full">
          <div className="absolute inset-0 flex flex-col gap-2 overflow-auto px-2 pr-2">
            {/* User Prompt - Moved to top for better dropdown space */}
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
                <UserPromptInputWithHistory
                  id="user_prompt"
                  value={querySettings.user_prompt || ''}
                  onChange={(value) => handleChange('user_prompt', value)}
                  onSelectFromHistory={handleSelectFromHistory}
                  onDeleteFromHistory={handleDeleteFromHistory}
                  history={userPromptHistory}
                  placeholder={t('retrievePanel.querySettings.userPromptPlaceholder')}
                  className="h-9"
                />
              </div>
            </>

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
              <div className="flex items-center gap-1">
                <Select
                  value={querySettings.mode}
                  onValueChange={(v) => handleChange('mode', v as QueryMode)}
                >
                  <SelectTrigger
                    id="query_mode_select"
                    className="hover:bg-primary/5 h-9 cursor-pointer focus:ring-0 focus:ring-offset-0 focus:outline-0 active:right-0 flex-1 text-left [&>span]:break-all [&>span]:line-clamp-1"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      <SelectItem value="naive">
                        {t('retrievePanel.querySettings.queryModeOptions.naive')}
                      </SelectItem>
                      <SelectItem value="local">
                        {t('retrievePanel.querySettings.queryModeOptions.local')}
                      </SelectItem>
                      <SelectItem value="global">
                        {t('retrievePanel.querySettings.queryModeOptions.global')}
                      </SelectItem>
                      <SelectItem value="hybrid">
                        {t('retrievePanel.querySettings.queryModeOptions.hybrid')}
                      </SelectItem>
                      <SelectItem value="mix">
                        {t('retrievePanel.querySettings.queryModeOptions.mix')}
                      </SelectItem>
                      <SelectItem value="bypass">
                        {t('retrievePanel.querySettings.queryModeOptions.bypass')}
                      </SelectItem>
                    </SelectGroup>
                  </SelectContent>
                </Select>
                <ResetButton onClick={() => handleReset('mode')} title="Reset to default (Mix)" />
              </div>
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
              <div className="flex items-center gap-1">
                <Input
                  id="top_k"
                  type="number"
                  value={querySettings.top_k ?? ''}
                  onChange={(e) => {
                    const value = e.target.value
                    handleChange('top_k', value === '' ? '' : Number.parseInt(value) || 0)
                  }}
                  onBlur={(e) => {
                    const value = e.target.value
                    if (value === '' || isNaN(Number.parseInt(value))) {
                      handleChange('top_k', 40)
                    }
                  }}
                  min={1}
                  placeholder={t('retrievePanel.querySettings.topKPlaceholder')}
                  className="h-9 flex-1 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                />
                <ResetButton onClick={() => handleReset('top_k')} title="Reset to default" />
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
              <div className="flex items-center gap-1">
                <Input
                  id="chunk_top_k"
                  type="number"
                  value={querySettings.chunk_top_k ?? ''}
                  onChange={(e) => {
                    const value = e.target.value
                    handleChange('chunk_top_k', value === '' ? '' : Number.parseInt(value) || 0)
                  }}
                  onBlur={(e) => {
                    const value = e.target.value
                    if (value === '' || isNaN(Number.parseInt(value))) {
                      handleChange('chunk_top_k', 20)
                    }
                  }}
                  min={1}
                  placeholder={t('retrievePanel.querySettings.chunkTopKPlaceholder')}
                  className="h-9 flex-1 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                />
                <ResetButton onClick={() => handleReset('chunk_top_k')} title="Reset to default" />
              </div>
            </>

            {/* Max Entity Tokens */}
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
              <div className="flex items-center gap-1">
                <Input
                  id="max_entity_tokens"
                  type="number"
                  value={querySettings.max_entity_tokens ?? ''}
                  onChange={(e) => {
                    const value = e.target.value
                    handleChange(
                      'max_entity_tokens',
                      value === '' ? '' : Number.parseInt(value) || 0
                    )
                  }}
                  onBlur={(e) => {
                    const value = e.target.value
                    if (value === '' || isNaN(Number.parseInt(value))) {
                      handleChange('max_entity_tokens', 6000)
                    }
                  }}
                  min={1}
                  placeholder={t('retrievePanel.querySettings.maxEntityTokensPlaceholder')}
                  className="h-9 flex-1 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                />
                <ResetButton
                  onClick={() => handleReset('max_entity_tokens')}
                  title="Reset to default"
                />
              </div>
            </>

            {/* Max Relation Tokens */}
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
              <div className="flex items-center gap-1">
                <Input
                  id="max_relation_tokens"
                  type="number"
                  value={querySettings.max_relation_tokens ?? ''}
                  onChange={(e) => {
                    const value = e.target.value
                    handleChange(
                      'max_relation_tokens',
                      value === '' ? '' : Number.parseInt(value) || 0
                    )
                  }}
                  onBlur={(e) => {
                    const value = e.target.value
                    if (value === '' || isNaN(Number.parseInt(value))) {
                      handleChange('max_relation_tokens', 8000)
                    }
                  }}
                  min={1}
                  placeholder={t('retrievePanel.querySettings.maxRelationTokensPlaceholder')}
                  className="h-9 flex-1 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                />
                <ResetButton
                  onClick={() => handleReset('max_relation_tokens')}
                  title="Reset to default"
                />
              </div>
            </>

            {/* Max Total Tokens */}
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
              <div className="flex items-center gap-1">
                <Input
                  id="max_total_tokens"
                  type="number"
                  value={querySettings.max_total_tokens ?? ''}
                  onChange={(e) => {
                    const value = e.target.value
                    handleChange(
                      'max_total_tokens',
                      value === '' ? '' : Number.parseInt(value) || 0
                    )
                  }}
                  onBlur={(e) => {
                    const value = e.target.value
                    if (value === '' || isNaN(Number.parseInt(value))) {
                      handleChange('max_total_tokens', 30000)
                    }
                  }}
                  min={1}
                  placeholder={t('retrievePanel.querySettings.maxTotalTokensPlaceholder')}
                  className="h-9 flex-1 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                />
                <ResetButton
                  onClick={() => handleReset('max_total_tokens')}
                  title="Reset to default"
                />
              </div>
            </>

            {/* Toggle Options */}
            <>
              <div className="flex items-center gap-2">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="enable_rerank" className="flex-1 ml-1 cursor-help">
                        {t('retrievePanel.querySettings.enableRerank')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.enableRerankTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <Checkbox
                  className="mr-10 cursor-pointer"
                  id="enable_rerank"
                  checked={querySettings.enable_rerank}
                  onCheckedChange={(checked) => handleChange('enable_rerank', checked)}
                />
              </div>

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
                  className="mr-10 cursor-pointer"
                  id="only_need_context"
                  checked={querySettings.only_need_context}
                  onCheckedChange={(checked) => {
                    handleChange('only_need_context', checked)
                    if (checked) {
                      handleChange('only_need_prompt', false)
                    }
                  }}
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
                  className="mr-10 cursor-pointer"
                  id="only_need_prompt"
                  checked={querySettings.only_need_prompt}
                  onCheckedChange={(checked) => {
                    handleChange('only_need_prompt', checked)
                    if (checked) {
                      handleChange('only_need_context', false)
                    }
                  }}
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
                  className="mr-10 cursor-pointer"
                  id="stream"
                  checked={querySettings.stream}
                  onCheckedChange={(checked) => handleChange('stream', checked)}
                />
              </div>
            </>

            {/* Citation Settings */}
            <>
              <div className="pt-2 mt-2 border-t border-gray-200 dark:border-gray-700">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="citation_mode_select" className="ml-1 cursor-help">
                        {t('retrievePanel.querySettings.citationMode')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.citationModeTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <div className="flex items-center gap-1">
                <Select
                  value={querySettings.citation_mode || 'none'}
                  onValueChange={(v) => handleChange('citation_mode', v as 'none' | 'inline' | 'footnotes')}
                >
                  <SelectTrigger
                    id="citation_mode_select"
                    className="hover:bg-primary/5 h-9 cursor-pointer focus:ring-0 focus:ring-offset-0 focus:outline-0 active:right-0 flex-1 text-left [&>span]:break-all [&>span]:line-clamp-1"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      <SelectItem value="none">
                        {t('retrievePanel.querySettings.citationModeOptions.none')}
                      </SelectItem>
                      <SelectItem value="inline">
                        {t('retrievePanel.querySettings.citationModeOptions.inline')}
                      </SelectItem>
                      <SelectItem value="footnotes">
                        {t('retrievePanel.querySettings.citationModeOptions.footnotes')}
                      </SelectItem>
                    </SelectGroup>
                  </SelectContent>
                </Select>
                <ResetButton onClick={() => handleReset('citation_mode')} title="Reset to default (None)" />
              </div>

              {/* Citation Threshold - only show when citation mode is not 'none' */}
              {querySettings.citation_mode && querySettings.citation_mode !== 'none' && (
                <>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <label htmlFor="citation_threshold" className="ml-1 cursor-help">
                          {t('retrievePanel.querySettings.citationThreshold')}
                        </label>
                      </TooltipTrigger>
                      <TooltipContent side="left">
                        <p>{t('retrievePanel.querySettings.citationThresholdTooltip')}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <div className="flex items-center gap-1">
                    <Input
                      id="citation_threshold"
                      type="number"
                      step="0.05"
                      min="0"
                      max="1"
                      value={querySettings.citation_threshold ?? 0.7}
                      onChange={(e) => {
                        const value = parseFloat(e.target.value)
                        if (!isNaN(value) && value >= 0 && value <= 1) {
                          handleChange('citation_threshold', value)
                        }
                      }}
                      className="h-9 flex-1 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                    />
                    <ResetButton onClick={() => handleReset('citation_threshold')} title="Reset to default (0.7)" />
                  </div>
                </>
              )}
            </>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
