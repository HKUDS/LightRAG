import { QueryPromptOverrides } from '@/api/lightrag'
import { pruneEmptyPromptOverrides } from './promptOverrides'

export const projectRetrievalVersionToOverrides = (
  payload?: Record<string, unknown> | null
): QueryPromptOverrides | undefined => {
  if (!payload || typeof payload !== 'object') {
    return undefined
  }

  const next: QueryPromptOverrides = {}
  const query = payload.query
  if (query && typeof query === 'object') {
    next.query = { ...(query as NonNullable<QueryPromptOverrides['query']>) }
  }

  const keywords = payload.keywords
  if (keywords && typeof keywords === 'object') {
    next.keywords = { ...(keywords as NonNullable<QueryPromptOverrides['keywords']>) }
  }

  return pruneEmptyPromptOverrides(next)
}
