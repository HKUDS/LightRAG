import type { QueryMode } from '@/api/lightrag'
import { isQueryEmpty, isRagQueryTooShort } from '@/utils/queryValidation'

export const SUPPORTED_QUERY_MODES: readonly QueryMode[] = [
  'naive',
  'local',
  'global',
  'hybrid',
  'mix',
  'bypass'
]

export type QueryInputError =
  | 'queryModePrefixInvalid'
  | 'queryModeError'
  | 'queryEmpty'
  | 'queryTooShort'

interface AcceptedQueryInput {
  ok: true
  query: string
  modeOverride?: QueryMode
}

interface RejectedQueryInput {
  ok: false
  error: QueryInputError
}

export type PreparedQueryInput = AcceptedQueryInput | RejectedQueryInput

/**
 * Parse the optional `/mode query` prefix and validate the effective query.
 * Both query entry points use this helper so they accept the same syntax and
 * apply the RAG-only minimum length to the prefix-stripped query.
 */
export function prepareQueryInput(input: string, defaultMode: QueryMode): PreparedQueryInput {
  const prefixMatch = input.match(/^\/(\w+)\s+([\s\S]+)/)

  if (/^\/\S+/.test(input) && !prefixMatch) {
    return { ok: false, error: 'queryModePrefixInvalid' }
  }

  let modeOverride: QueryMode | undefined
  let query = input

  if (prefixMatch) {
    const requestedMode = prefixMatch[1]
    if (!SUPPORTED_QUERY_MODES.includes(requestedMode as QueryMode)) {
      return { ok: false, error: 'queryModeError' }
    }
    modeOverride = requestedMode as QueryMode
    query = prefixMatch[2]
  }

  // A prefix can swallow the whole input ('/bypass  '), so the effective
  // query is checked for emptiness on every mode, bypass included.
  if (isQueryEmpty(query)) {
    return { ok: false, error: 'queryEmpty' }
  }

  if (isRagQueryTooShort(query, modeOverride ?? defaultMode)) {
    return { ok: false, error: 'queryTooShort' }
  }

  return { ok: true, query, modeOverride }
}
