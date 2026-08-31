import { useCallback } from 'react'
import { useSettingsStore } from '@/stores/settings'
import { useQuerySettingsStore } from '@/stores/querySettings'
import { useWebuiRetrievalHistoryStore } from '@/stores/webuiRetrievalHistory'
import QuerySettings from '@/components/retrieval/QuerySettings'
import MessageList from '@/features/retrieval/MessageList'
import QueryComposer from '@/features/retrieval/QueryComposer'
import { useQuerySession } from '@/features/retrieval/useQuerySession'
import { useTranslation } from 'react-i18next'
import { prepareQueryInput, SUPPORTED_QUERY_MODES } from '@/features/retrieval/queryInput'

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

  // A leading `/mode ` prefix overrides the query mode for this one request.
  // The workspace entry uses the same shared preparation contract.
  const handleSend = useCallback(
    (input: string): string | null => {
      const prepared = prepareQueryInput(input, getQuerySettingsSnapshot().mode)
      if (!prepared.ok) {
        return t(`retrievePanel.retrieval.${prepared.error}`, {
          modes: SUPPORTED_QUERY_MODES.join(', ')
        })
      }

      // The displayed user message keeps the original input (prefix and all);
      // the request carries the stripped query — same as before extraction.
      void session.submitQuery(prepared.query, {
        modeOverride: prepared.modeOverride,
        displayedInput: input
      })
      return null
    },
    [getQuerySettingsSnapshot, session, t]
  )

  return (
    <div className="flex size-full gap-2 overflow-hidden px-2 pb-12">
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
