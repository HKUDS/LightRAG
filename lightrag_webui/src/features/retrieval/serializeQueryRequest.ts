import type { QueryRequest } from '@/api/lightrag'
import type { QuerySettings } from '@/stores/querySettings'
import type { MessageWithError } from '@/types/retrieval'

/**
 * Turns a bypass-mode query with no configured history turns into a
 * 3-turn conversation (existing behavior, kept verbatim).
 */
export const BYPASS_DEFAULT_HISTORY_TURNS = 3

/**
 * The single request serializer shared by BOTH query pages.
 *
 * EQUIVALENCE CONTRACT: given the same `querySettings` snapshot, the same
 * explicitly-passed history and the same question, this function produces an
 * IDENTICAL request body — it contains no entry-dependent branches, no mode
 * clamping, no field filtering and no history special-casing per entry.
 * All page differences live in the inputs:
 * - the settings snapshot is BUILT by the page composition layer (the
 *   workspace clamps its two debug-only switches to false there;
 *   the admin page applies its `/mode` prefix override there);
 * - the history is each page's OWN store content, passed explicitly.
 *
 * `history_turns` is a client-side knob (how many turns to expand into
 * `conversation_history`); the server's QueryRequest model never declared it
 * and silently dropped it, so it is stripped from the request body after
 * expansion.
 */
export function serializeQueryRequest(
  querySettings: QuerySettings,
  query: string,
  history: MessageWithError[]
): QueryRequest {
  const configuredHistoryTurns = querySettings.history_turns || 0
  const effectiveHistoryTurns =
    querySettings.mode === 'bypass' && configuredHistoryTurns === 0
      ? BYPASS_DEFAULT_HISTORY_TURNS
      : configuredHistoryTurns

  const request: QueryRequest = {
    ...querySettings,
    query,
    response_type: 'Multiple Paragraphs',
    // Request retrieval progress events for the live progress display.
    include_progress: true,
    conversation_history:
      effectiveHistoryTurns > 0
        ? history
          .filter((m) => m.isError !== true)
          .slice(-effectiveHistoryTurns * 2)
          .map((m) => ({ role: m.role, content: m.content }))
        : []
  }

  delete request.history_turns
  return request
}
