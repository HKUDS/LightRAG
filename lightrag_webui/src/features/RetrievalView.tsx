import { useCallback } from 'react'
import { useSettingsStore } from '@/stores/settings'
import { useQuerySettingsStore } from '@/stores/querySettings'
import { useWebuiRetrievalHistoryStore } from '@/stores/webuiRetrievalHistory'
import QuerySettings from '@/components/retrieval/QuerySettings'
import MessageList from '@/features/retrieval/MessageList'
import QueryComposer from '@/features/retrieval/QueryComposer'
import { useQuerySession } from '@/features/retrieval/useQuerySession'
import { useTranslation } from 'react-i18next'
import type { QueryMode } from '@/api/lightrag'
import { isRagQueryTooShort } from '@/utils/queryValidation'

const allowedModes: QueryMode[] = ['naive', 'local', 'global', 'hybrid', 'mix', 'bypass']

/**
 * The ADMIN query page: composes the shared query session, message list and
 * composer with the QuerySettings sidebar and the `/mode` prefix override.
 * Its history storage is the admin entry's own store; the settings snapshot
 * is passed through unmodified (including the `only_need_*` debug switches —
 * those are clamped only on the workspace page).
 */
export default function RetrievalView() {
  const { t } = useTranslation()
  // Get current tab to determine if this tab is active (for performance optimization)
  const currentTab = useSettingsStore.use.currentTab()
  const isRetrievalTabActive = currentTab === 'retrieval'

  const getQuerySettingsSnapshot = useCallback(
    () => useQuerySettingsStore.getState().querySettings,
    []
  )

  const onUserPromptUsed = useCallback((prompt: string) => {
    useSettingsStore.getState().addUserPromptToHistory(prompt)
  }, [])

  const session = useQuerySession({
    historyStore: useWebuiRetrievalHistoryStore,
    getQuerySettingsSnapshot,
    onUserPromptUsed
  })

  // Admin-only input convention: a leading `/mode ` prefix overrides the
  // query mode for this one request. This parsing is the ONLY request-
  // serialization difference between the two query pages.
  const handleSend = useCallback(
    (input: string): string | null => {
      const prefixMatch = input.match(/^\/(\w+)\s+([\s\S]+)/)

      // If input starts with a slash, but does not match the valid prefix
      // pattern, treat as error
      if (/^\/\S+/.test(input) && !prefixMatch) {
        return t('retrievePanel.retrieval.queryModePrefixInvalid')
      }

      let modeOverride: QueryMode | undefined
      let actualQuery = input
      if (prefixMatch) {
        const mode = prefixMatch[1] as QueryMode
        if (!allowedModes.includes(mode)) {
          return t('retrievePanel.retrieval.queryModeError', {
            modes: 'naive, local, global, hybrid, mix, bypass'
          })
        }
        modeOverride = mode
        actualQuery = prefixMatch[2]
      }

      const effectiveMode = modeOverride ?? getQuerySettingsSnapshot().mode
      if (isRagQueryTooShort(actualQuery, effectiveMode)) {
        return t('retrievePanel.retrieval.queryTooShort')
      }

      // The displayed user message keeps the original input (prefix and all);
      // the request carries the stripped query — same as before extraction.
      void session.submitQuery(actualQuery, { modeOverride, displayedInput: input })
      return null
    },
    [getQuerySettingsSnapshot, session, t]
  )

  return (
    <div className="flex size-full gap-2 px-2 pb-12 overflow-hidden">
      <div className="flex grow flex-col gap-4">
        <MessageList
          messages={session.messages}
          isLoading={session.isLoading}
          queryProgress={session.queryProgress}
          isActive={isRetrievalTabActive}
          emptyState={
            <div className="text-muted-foreground text-lg">
              {t('retrievePanel.retrieval.startPrompt')}
            </div>
          }
        />
        <QueryComposer
          isLoading={session.isLoading}
          stopDisabled={session.stopDisabled}
          onSend={handleSend}
          onStop={session.handleStop}
          onClear={session.clearMessages}
        />
      </div>
      <QuerySettings />
    </div>
  )
}
