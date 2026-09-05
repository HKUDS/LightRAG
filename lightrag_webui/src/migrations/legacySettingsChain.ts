/**
 * The legacy `settings-storage` v1 → v21 migration chain, extracted from
 * `stores/settings.ts` into a side-effect-free pure module.
 *
 * It is imported by BOTH:
 * - the one-time storage-split migrator (`splitSettingsStorage.ts`), which
 *   must normalize any envelope <= v21 through this exact chain before
 *   splitting fields out (reusing — not copying — the chain), and
 * - the settings store's Zustand `persist.migrate`.
 *
 * PURITY CONTRACT: this module must not import any store, `App`, the API
 * client, or the navigation service — importing it must not create or
 * hydrate a persist store, and must not read or write localStorage. The
 * split migrator runs before any store module is evaluated; a store import
 * here would defeat that ordering.
 */

import { defaultQueryLabel, suggestedUserPrompts } from '@/lib/queryDefaults'

/**
 * Highest legacy envelope version this chain normalizes to. Envelopes with a
 * larger version (rollback-then-upgrade, or unknown future versions) must be
 * left completely untouched by the migrator: not read, not cleaned, not
 * overwritten.
 */
export const LEGACY_SETTINGS_VERSION_CAP = 21

/**
 * Normalize a legacy persisted settings state from `version` up to v21.
 * Mutates and returns `state` (same contract as the original inline chain).
 */
export function migrateLegacySettingsState(state: any, version: number): any {
  if (version < 2) {
    state.showEdgeLabel = false
  }
  if (version < 3) {
    state.queryLabel = defaultQueryLabel
  }
  if (version < 4) {
    state.showPropertyPanel = true
    state.showNodeSearchBar = true
    state.showNodeLabel = true
    state.enableHealthCheck = true
    state.apiKey = null
  }
  if (version < 5) {
    state.currentTab = 'documents'
  }
  if (version < 6) {
    state.querySettings = {
      mode: 'global',
      response_type: 'Multiple Paragraphs',
      top_k: 10,
      max_token_for_text_unit: 4000,
      max_token_for_global_context: 4000,
      max_token_for_local_context: 4000,
      only_need_context: false,
      only_need_prompt: false,
      stream: true,
      history_turns: 0,
      hl_keywords: [],
      ll_keywords: []
    }
    state.retrievalHistory = []
  }
  if (version < 7) {
    state.graphQueryMaxDepth = 3
  }
  if (version < 8) {
    state.graphMinDegree = 0
    state.language = 'en'
  }
  if (version < 9) {
    state.showFileName = false
  }
  if (version < 10) {
    delete state.graphMinDegree // 删除废弃参数
    state.graphMaxNodes = 1000 // 添加新参数
  }
  if (version < 11) {
    state.minEdgeSize = 1
    state.maxEdgeSize = 1
  }
  if (version < 12) {
    // Clear retrieval history to avoid compatibility issues with MessageWithError type
    state.retrievalHistory = []
  }
  if (version < 13) {
    // Add user_prompt field for older versions
    if (state.querySettings) {
      state.querySettings.user_prompt = ''
    }
  }
  if (version < 14) {
    // Add backendMaxGraphNodes field for older versions
    state.backendMaxGraphNodes = null
  }
  if (version < 15) {
    // Add new querySettings
    state.querySettings = {
      ...state.querySettings,
      mode: 'mix',
      response_type: 'Multiple Paragraphs',
      top_k: 40,
      chunk_top_k: 10,
      max_entity_tokens: 10000,
      max_relation_tokens: 10000,
      max_total_tokens: 32000,
      enable_rerank: true,
      history_turns: 0
    }
  }
  if (version < 16) {
    // Add documentsPageSize field for older versions
    state.documentsPageSize = 10
  }
  if (version < 17) {
    // Force history_turns to 0 for all users
    if (state.querySettings) {
      state.querySettings.history_turns = 0
    }
  }
  if (version < 18) {
    // Add userPromptHistory field for older versions
    state.userPromptHistory = []
  }
  if (version < 19) {
    // Remove deprecated response_type parameter
    if (state.querySettings) {
      delete state.querySettings.response_type
    }
  }
  if (version < 20) {
    // One-time injection of system-suggested prompts; append after any existing
    // history so user's own prompts keep priority. Skip any prompt already
    // present to avoid duplicate dropdown entries (matches the de-duping in
    // addUserPromptToHistory). version monotonicity makes this run at most once.
    const existing = state.userPromptHistory ?? []
    state.userPromptHistory = [
      ...existing,
      ...suggestedUserPrompts.filter((p: string) => !existing.includes(p))
    ]
  }
  if (version < 21) {
    // The API tab was replaced by a header icon that opens /docs in a
    // new browser tab; a persisted 'api' would select a tab that no
    // longer exists and leave the content area empty.
    if (state.currentTab === 'api') {
      state.currentTab = 'documents'
    }
  }
  return state
}
