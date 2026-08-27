import { useCallback } from 'react'
import { useQuerySettingsStore } from '@/stores/querySettings'
import { useWorkspaceRetrievalHistoryStore } from '@/stores/workspaceRetrievalHistory'
import MessageList from '@/features/retrieval/MessageList'
import QueryComposer from '@/features/retrieval/QueryComposer'
import { useQuerySession } from '@/features/retrieval/useQuerySession'
import WorkspaceEmptyState from './WorkspaceEmptyState'
import type { QuerySettings } from '@/stores/querySettings'

/**
 * The workspace (query-user) page. Composes the SAME shared query session,
 * message list and composer as the admin `RetrievalView`, with these
 * page-level differences (and only these):
 * - no QuerySettings sidebar and no `/mode` prefix parsing: a question
 *   starting with '/' is submitted as plain text;
 * - the settings snapshot inherits the browser-shared `querySettings` saved
 *   by /webui, read-only, EXCEPT `only_need_context`/`only_need_prompt`
 *   which are clamped to false HERE, at the composition layer (they are
 *   debug outlets that make the server not answer; the shared serializer
 *   knows nothing about this clamping);
 * - its history is the workspace entry's own store;
 * - the message area is always active (no admin tab lifecycle).
 */
export default function WorkspaceQueryView() {
  const getQuerySettingsSnapshot = useCallback((): QuerySettings => {
    const settings = useQuerySettingsStore.getState().querySettings
    return {
      ...settings,
      only_need_context: false,
      only_need_prompt: false
    }
  }, [])

  const session = useQuerySession({
    historyStore: useWorkspaceRetrievalHistoryStore,
    getQuerySettingsSnapshot
    // No onUserPromptUsed: the inherited user_prompt is not recorded into the
    // admin prompt-history from this entry (that would write settings-storage).
  })

  // No prefix parsing: submit the input verbatim as the question.
  const handleSend = useCallback(
    (input: string): string | null => {
      void session.submitQuery(input)
      return null
    },
    [session]
  )

  return (
    <div className="flex size-full flex-col gap-2 overflow-hidden px-2 pb-2 pt-2 md:gap-4 md:px-4 md:pb-4">
      <MessageList
        messages={session.messages}
        isLoading={session.isLoading}
        queryProgress={session.queryProgress}
        isActive={true}
        emptyState={<WorkspaceEmptyState />}
      />
      <QueryComposer
        isLoading={session.isLoading}
        stopDisabled={session.stopDisabled}
        onSend={handleSend}
        onStop={session.handleStop}
        onClear={session.clearMessages}
      />
    </div>
  )
}
