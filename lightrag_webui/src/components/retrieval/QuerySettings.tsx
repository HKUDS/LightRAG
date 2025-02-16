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

export default function QuerySettings() {
  const querySettings = useSettingsStore((state) => state.querySettings)

  const handleChange = useCallback((key: keyof QueryRequest, value: any) => {
    useSettingsStore.getState().updateQuerySettings({ [key]: value })
  }, [])

  return (
    <Card className="flex shrink-0 flex-col">
      <CardHeader className="px-4 pt-4 pb-2">
        <CardTitle>Parameters</CardTitle>
        <CardDescription>Configure your query parameters</CardDescription>
      </CardHeader>
      <CardContent className="m-0 flex grow flex-col p-0 text-xs">
        <div className="relative size-full">
          <div className="absolute inset-0 flex flex-col gap-2 overflow-auto px-2">
            {/* Query Mode */}
            <>
              <Text
                className="ml-1"
                text="Query Mode"
                tooltip="Select the retrieval strategy:\n• Naive: Basic search without advanced techniques\n• Local: Context-dependent information retrieval\n• Global: Utilizes global knowledge base\n• Hybrid: Combines local and global retrieval\n• Mix: Integrates knowledge graph with vector retrieval"
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
                    <SelectItem value="naive">Naive</SelectItem>
                    <SelectItem value="local">Local</SelectItem>
                    <SelectItem value="global">Global</SelectItem>
                    <SelectItem value="hybrid">Hybrid</SelectItem>
                    <SelectItem value="mix">Mix</SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
            </>

            {/* Response Format */}
            <>
              <Text
                className="ml-1"
                text="Response Format"
                tooltip="Defines the response format. Examples:\n• Multiple Paragraphs\n• Single Paragraph\n• Bullet Points"
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
                    <SelectItem value="Multiple Paragraphs">Multiple Paragraphs</SelectItem>
                    <SelectItem value="Single Paragraph">Single Paragraph</SelectItem>
                    <SelectItem value="Bullet Points">Bullet Points</SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
            </>

            {/* Top K */}
            <>
              <Text
                className="ml-1"
                text="Top K Results"
                tooltip="Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode"
                side="left"
              />
              <NumberInput
                id="top_k"
                stepper={1}
                value={querySettings.top_k}
                onValueChange={(v) => handleChange('top_k', v)}
                min={1}
                placeholder="Number of results"
              />
            </>

            {/* Max Tokens */}
            <>
              <>
                <Text
                  className="ml-1"
                  text="Max Tokens for Text Unit"
                  tooltip="Maximum number of tokens allowed for each retrieved text chunk"
                  side="left"
                />
                <NumberInput
                  id="max_token_for_text_unit"
                  stepper={500}
                  value={querySettings.max_token_for_text_unit}
                  onValueChange={(v) => handleChange('max_token_for_text_unit', v)}
                  min={1}
                  placeholder="Max tokens for text unit"
                />
              </>

              <>
                <Text
                  text="Max Tokens for Global Context"
                  tooltip="Maximum number of tokens allocated for relationship descriptions in global retrieval"
                  side="left"
                />
                <NumberInput
                  id="max_token_for_global_context"
                  stepper={500}
                  value={querySettings.max_token_for_global_context}
                  onValueChange={(v) => handleChange('max_token_for_global_context', v)}
                  min={1}
                  placeholder="Max tokens for global context"
                />
              </>

              <>
                <Text
                  className="ml-1"
                  text="Max Tokens for Local Context"
                  tooltip="Maximum number of tokens allocated for entity descriptions in local retrieval"
                  side="left"
                />
                <NumberInput
                  id="max_token_for_local_context"
                  stepper={500}
                  value={querySettings.max_token_for_local_context}
                  onValueChange={(v) => handleChange('max_token_for_local_context', v)}
                  min={1}
                  placeholder="Max tokens for local context"
                />
              </>
            </>

            {/* History Turns */}
            <>
              <Text
                className="ml-1"
                text="History Turns"
                tooltip="Number of complete conversation turns (user-assistant pairs) to consider in the response context"
                side="left"
              />
              <NumberInput
                className="!border-input"
                id="history_turns"
                stepper={1}
                type="text"
                value={querySettings.history_turns}
                onValueChange={(v) => handleChange('history_turns', v)}
                min={0}
                placeholder="Number of history turns"
              />
            </>

            {/* Keywords */}
            <>
              <>
                <Text
                  className="ml-1"
                  text="High-Level Keywords"
                  tooltip="List of high-level keywords to prioritize in retrieval. Separate with commas"
                  side="left"
                />
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
                  placeholder="Enter keywords"
                />
              </>

              <>
                <Text
                  className="ml-1"
                  text="Low-Level Keywords"
                  tooltip="List of low-level keywords to refine retrieval focus. Separate with commas"
                  side="left"
                />
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
                  placeholder="Enter keywords"
                />
              </>
            </>

            {/* Toggle Options */}
            <>
              <div className="flex items-center gap-2">
                <Text
                  className="ml-1"
                  text="Only Need Context"
                  tooltip="If True, only returns the retrieved context without generating a response"
                  side="left"
                />
                <div className="grow" />
                <Checkbox
                  className="mr-1 cursor-pointer"
                  id="only_need_context"
                  checked={querySettings.only_need_context}
                  onCheckedChange={(checked) => handleChange('only_need_context', checked)}
                />
              </div>

              <div className="flex items-center gap-2">
                <Text
                  className="ml-1"
                  text="Only Need Prompt"
                  tooltip="If True, only returns the generated prompt without producing a response"
                  side="left"
                />
                <div className="grow" />
                <Checkbox
                  className="mr-1 cursor-pointer"
                  id="only_need_prompt"
                  checked={querySettings.only_need_prompt}
                  onCheckedChange={(checked) => handleChange('only_need_prompt', checked)}
                />
              </div>

              <div className="flex items-center gap-2">
                <Text
                  className="ml-1"
                  text="Stream Response"
                  tooltip="If True, enables streaming output for real-time responses"
                  side="left"
                />
                <div className="grow" />
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
