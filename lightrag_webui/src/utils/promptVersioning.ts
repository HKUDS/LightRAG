import { PromptConfigGroup, PromptVersionRecord, QueryPromptOverrides } from '@/api/lightrag'
import { pruneEmptyPromptOverrides } from './promptOverrides'

export type PromptEditorFieldType = 'input' | 'textarea' | 'list'

export type PromptEditorSection = {
  key: string
  title: string
  type: PromptEditorFieldType
}

const INDEXING_EDITOR_SECTIONS: PromptEditorSection[] = [
  { key: 'entity_types', title: 'ENTITY_TYPES', type: 'list' },
  { key: 'summary_language', title: 'SUMMARY_LANGUAGE', type: 'input' },
  { key: 'shared.tuple_delimiter', title: 'shared.tuple_delimiter', type: 'input' },
  { key: 'shared.completion_delimiter', title: 'shared.completion_delimiter', type: 'input' },
  { key: 'entity_extraction.system_prompt', title: 'entity_extraction.system_prompt', type: 'textarea' },
  { key: 'entity_extraction.user_prompt', title: 'entity_extraction.user_prompt', type: 'textarea' },
  { key: 'entity_extraction.continue_prompt', title: 'entity_extraction.continue_prompt', type: 'textarea' },
  { key: 'entity_extraction.examples', title: 'entity_extraction.examples', type: 'list' },
  { key: 'summary.summarize_entity_descriptions', title: 'summary.summarize_entity_descriptions', type: 'textarea' }
]

const RETRIEVAL_EDITOR_SECTIONS: PromptEditorSection[] = [
  { key: 'query.rag_response', title: 'query.rag_response', type: 'textarea' },
  { key: 'query.naive_rag_response', title: 'query.naive_rag_response', type: 'textarea' },
  { key: 'query.kg_query_context', title: 'query.kg_query_context', type: 'textarea' },
  { key: 'query.naive_query_context', title: 'query.naive_query_context', type: 'textarea' },
  { key: 'keywords.keywords_extraction', title: 'keywords.keywords_extraction', type: 'textarea' },
  { key: 'keywords.keywords_extraction_examples', title: 'keywords.keywords_extraction_examples', type: 'list' }
]

export const buildPromptEditorSections = (group: PromptConfigGroup): PromptEditorSection[] =>
  group === 'indexing' ? INDEXING_EDITOR_SECTIONS : RETRIEVAL_EDITOR_SECTIONS

export const formatVersionLineageLabel = (
  version: Pick<PromptVersionRecord, 'source_version_id'>,
  versionsById: Record<string, PromptVersionRecord>
): string => {
  if (!version.source_version_id) {
    return 'Manual'
  }
  return versionsById[version.source_version_id]?.version_name || 'Deleted'
}

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
