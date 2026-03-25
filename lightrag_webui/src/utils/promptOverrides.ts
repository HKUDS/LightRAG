import { QueryPromptOverrides } from '@/api/lightrag'

const cleanString = (value?: string): string | undefined => {
  if (typeof value !== 'string') {
    return undefined
  }
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : undefined
}

export const setExampleItem = (items: string[], index: number, value: string): string[] => {
  const next = [...items]
  while (next.length <= index) {
    next.push('')
  }
  next[index] = value
  return next.map((item) => item.trim()).filter((item) => item.length > 0)
}

export const pruneEmptyPromptOverrides = (
  overrides?: QueryPromptOverrides
): QueryPromptOverrides | undefined => {
  if (!overrides) {
    return undefined
  }

  const query = {
    rag_response: cleanString(overrides.query?.rag_response),
    naive_rag_response: cleanString(overrides.query?.naive_rag_response),
    kg_query_context: cleanString(overrides.query?.kg_query_context),
    naive_query_context: cleanString(overrides.query?.naive_query_context)
  }

  const keywordsExtraction = cleanString(overrides.keywords?.keywords_extraction)
  const keywordsExamples = (overrides.keywords?.keywords_extraction_examples || [])
    .map((item) => item.trim())
    .filter((item) => item.length > 0)

  const next: QueryPromptOverrides = {}

  if (Object.values(query).some(Boolean)) {
    next.query = {}
    if (query.rag_response) {
      next.query.rag_response = query.rag_response
    }
    if (query.naive_rag_response) {
      next.query.naive_rag_response = query.naive_rag_response
    }
    if (query.kg_query_context) {
      next.query.kg_query_context = query.kg_query_context
    }
    if (query.naive_query_context) {
      next.query.naive_query_context = query.naive_query_context
    }
  }

  if (keywordsExtraction || keywordsExamples.length > 0) {
    next.keywords = {}
    if (keywordsExtraction) {
      next.keywords.keywords_extraction = keywordsExtraction
    }
    if (keywordsExamples.length > 0) {
      next.keywords.keywords_extraction_examples = keywordsExamples
    }
  }

  return Object.keys(next).length > 0 ? next : undefined
}
