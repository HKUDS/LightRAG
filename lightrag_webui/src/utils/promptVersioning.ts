import { PromptConfigGroup, PromptVersionRecord, QueryPromptOverrides } from '@/api/lightrag'
import { pruneEmptyPromptOverrides } from './promptOverrides'

export type PromptEditorFieldType = 'input' | 'textarea' | 'list' | 'csv'

export type PromptEditorVariable = {
  label: string
  descriptionKey: string
}

export type PromptEditorSection = {
  key: string
  title: string
  type: PromptEditorFieldType
  helpKey: string
  descriptionKey: string
  variables: PromptEditorVariable[]
  itemPlaceholderKey?: string
}

const variable = (label: string, descriptionKey: string): PromptEditorVariable => ({
  label,
  descriptionKey
})

const INDEXING_EDITOR_SECTIONS: PromptEditorSection[] = [
  {
    key: 'entity_types',
    title: 'ENTITY_TYPES',
    type: 'csv',
    helpKey: 'entityTypes',
    descriptionKey: 'promptManagement.fieldHelp.entityTypes.description',
    variables: [variable('{entity_types}', 'promptManagement.variables.entityTypes')],
    itemPlaceholderKey: 'promptManagement.fieldHelp.entityTypes.itemPlaceholder'
  },
  {
    key: 'summary_language',
    title: 'SUMMARY_LANGUAGE',
    type: 'input',
    helpKey: 'summaryLanguage',
    descriptionKey: 'promptManagement.fieldHelp.summaryLanguage.description',
    variables: [variable('{language}', 'promptManagement.variables.language')]
  },
  {
    key: 'shared.tuple_delimiter',
    title: 'shared.tuple_delimiter',
    type: 'input',
    helpKey: 'tupleDelimiter',
    descriptionKey: 'promptManagement.fieldHelp.tupleDelimiter.description',
    variables: [variable('{tuple_delimiter}', 'promptManagement.variables.tupleDelimiter')]
  },
  {
    key: 'shared.completion_delimiter',
    title: 'shared.completion_delimiter',
    type: 'input',
    helpKey: 'completionDelimiter',
    descriptionKey: 'promptManagement.fieldHelp.completionDelimiter.description',
    variables: [variable('{completion_delimiter}', 'promptManagement.variables.completionDelimiter')]
  },
  {
    key: 'entity_extraction.system_prompt',
    title: 'entity_extraction.system_prompt',
    type: 'textarea',
    helpKey: 'entitySystemPrompt',
    descriptionKey: 'promptManagement.fieldHelp.entitySystemPrompt.description',
    variables: [
      variable('{tuple_delimiter}', 'promptManagement.variables.tupleDelimiter'),
      variable('{completion_delimiter}', 'promptManagement.variables.completionDelimiter'),
      variable('{entity_types}', 'promptManagement.variables.entityTypes'),
      variable('{language}', 'promptManagement.variables.language'),
      variable('{examples}', 'promptManagement.variables.examples')
    ]
  },
  {
    key: 'entity_extraction.user_prompt',
    title: 'entity_extraction.user_prompt',
    type: 'textarea',
    helpKey: 'entityUserPrompt',
    descriptionKey: 'promptManagement.fieldHelp.entityUserPrompt.description',
    variables: [
      variable('{input_text}', 'promptManagement.variables.inputText'),
      variable('{completion_delimiter}', 'promptManagement.variables.completionDelimiter'),
      variable('{language}', 'promptManagement.variables.language'),
      variable('{entity_types}', 'promptManagement.variables.entityTypes')
    ]
  },
  {
    key: 'entity_extraction.continue_prompt',
    title: 'entity_extraction.continue_prompt',
    type: 'textarea',
    helpKey: 'entityContinuePrompt',
    descriptionKey: 'promptManagement.fieldHelp.entityContinuePrompt.description',
    variables: [
      variable('{tuple_delimiter}', 'promptManagement.variables.tupleDelimiter'),
      variable('{completion_delimiter}', 'promptManagement.variables.completionDelimiter'),
      variable('{language}', 'promptManagement.variables.language')
    ]
  },
  {
    key: 'entity_extraction.examples',
    title: 'entity_extraction.examples',
    type: 'list',
    helpKey: 'entityExamples',
    descriptionKey: 'promptManagement.fieldHelp.entityExamples.description',
    variables: [
      variable('{examples}', 'promptManagement.variables.examples'),
      variable('{entity_types}', 'promptManagement.variables.entityTypes'),
      variable('{tuple_delimiter}', 'promptManagement.variables.tupleDelimiter'),
      variable('{completion_delimiter}', 'promptManagement.variables.completionDelimiter')
    ],
    itemPlaceholderKey: 'promptManagement.fieldHelp.entityExamples.itemPlaceholder'
  },
  {
    key: 'summary.summarize_entity_descriptions',
    title: 'summary.summarize_entity_descriptions',
    type: 'textarea',
    helpKey: 'summaryDescriptions',
    descriptionKey: 'promptManagement.fieldHelp.summaryDescriptions.description',
    variables: [
      variable('{description_list}', 'promptManagement.variables.descriptionList'),
      variable('{description_type}', 'promptManagement.variables.descriptionType'),
      variable('{description_name}', 'promptManagement.variables.descriptionName'),
      variable('{summary_length}', 'promptManagement.variables.summaryLength'),
      variable('{language}', 'promptManagement.variables.language')
    ]
  }
]

const RETRIEVAL_EDITOR_SECTIONS: PromptEditorSection[] = [
  {
    key: 'query.rag_response',
    title: 'query.rag_response',
    type: 'textarea',
    helpKey: 'ragResponse',
    descriptionKey: 'promptManagement.fieldHelp.ragResponse.description',
    variables: [
      variable('{context_data}', 'promptManagement.variables.contextData'),
      variable('{response_type}', 'promptManagement.variables.responseType'),
      variable('{user_prompt}', 'promptManagement.variables.userPrompt')
    ]
  },
  {
    key: 'query.naive_rag_response',
    title: 'query.naive_rag_response',
    type: 'textarea',
    helpKey: 'naiveRagResponse',
    descriptionKey: 'promptManagement.fieldHelp.naiveRagResponse.description',
    variables: [
      variable('{content_data}', 'promptManagement.variables.contentData'),
      variable('{response_type}', 'promptManagement.variables.responseType'),
      variable('{user_prompt}', 'promptManagement.variables.userPrompt')
    ]
  },
  {
    key: 'query.kg_query_context',
    title: 'query.kg_query_context',
    type: 'textarea',
    helpKey: 'kgQueryContext',
    descriptionKey: 'promptManagement.fieldHelp.kgQueryContext.description',
    variables: [
      variable('{entities_str}', 'promptManagement.variables.entitiesStr'),
      variable('{relations_str}', 'promptManagement.variables.relationsStr'),
      variable('{text_chunks_str}', 'promptManagement.variables.textChunksStr'),
      variable('{reference_list_str}', 'promptManagement.variables.referenceListStr')
    ]
  },
  {
    key: 'query.naive_query_context',
    title: 'query.naive_query_context',
    type: 'textarea',
    helpKey: 'naiveQueryContext',
    descriptionKey: 'promptManagement.fieldHelp.naiveQueryContext.description',
    variables: [
      variable('{text_chunks_str}', 'promptManagement.variables.textChunksStr'),
      variable('{reference_list_str}', 'promptManagement.variables.referenceListStr')
    ]
  },
  {
    key: 'keywords.keywords_extraction',
    title: 'keywords.keywords_extraction',
    type: 'textarea',
    helpKey: 'keywordsExtraction',
    descriptionKey: 'promptManagement.fieldHelp.keywordsExtraction.description',
    variables: [
      variable('{query}', 'promptManagement.variables.query'),
      variable('{examples}', 'promptManagement.variables.examples'),
      variable('{language}', 'promptManagement.variables.language')
    ]
  },
  {
    key: 'keywords.keywords_extraction_examples',
    title: 'keywords.keywords_extraction_examples',
    type: 'list',
    helpKey: 'keywordsExtractionExamples',
    descriptionKey: 'promptManagement.fieldHelp.keywordsExtractionExamples.description',
    variables: [variable('{examples}', 'promptManagement.variables.examples')],
    itemPlaceholderKey: 'promptManagement.fieldHelp.keywordsExtractionExamples.itemPlaceholder'
  }
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

export const getPromptFieldPreview = (value: unknown): string => {
  if (Array.isArray(value)) {
    return value
      .map((item) => String(item).trim())
      .filter(Boolean)
      .join(', ')
  }

  if (typeof value === 'string') {
    return value.split('\n')[0].trim()
  }

  if (value == null) {
    return ''
  }

  return String(value)
}

export const getPromptFieldEditorValue = (
  section: Pick<PromptEditorSection, 'type' | 'key'>,
  value: unknown
): string => {
  if (section.type === 'csv' || section.key === 'entity_types') {
    if (Array.isArray(value)) {
      return value
        .map((item) => String(item).trim())
        .filter(Boolean)
        .join(',')
    }
  }

  return typeof value === 'string' ? value : ''
}

const getLocaleDefaultVersion = (
  versions: PromptVersionRecord[],
  groupType: PromptConfigGroup,
  locale: string
): PromptVersionRecord | undefined =>
  versions.find((version) => version.version_name === `${groupType}-${locale}-default`)

const isSeedDefaultVersion = (
  version: PromptVersionRecord,
  groupType: PromptConfigGroup
): boolean => version.version_name.startsWith(`${groupType}-`) && version.version_name.endsWith('-default')

export type PromptVersionSelectionMode = 'automatic' | 'manual'

export const getPreferredPromptVersionId = ({
  versions,
  activeVersionId,
  selectedVersionId,
  groupType,
  locale,
  selectionMode = 'automatic'
}: {
  versions: PromptVersionRecord[]
  activeVersionId: string | null
  selectedVersionId: string | null
  groupType: PromptConfigGroup
  locale: string
  selectionMode?: PromptVersionSelectionMode
}): string | null => {
  const localeDefaultVersion = getLocaleDefaultVersion(versions, groupType, locale)
  const selectedVersion = versions.find((version) => version.version_id === selectedVersionId) || null

  if (
    selectionMode === 'automatic' &&
    selectedVersion &&
    localeDefaultVersion &&
    isSeedDefaultVersion(selectedVersion, groupType) &&
    selectedVersion.version_id !== localeDefaultVersion.version_id
  ) {
    return localeDefaultVersion.version_id
  }

  if (selectedVersion) {
    return selectedVersion.version_id
  }

  return localeDefaultVersion?.version_id || activeVersionId || versions[0]?.version_id || null
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
