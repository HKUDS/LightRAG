/**
 * Pure query-related defaults.
 *
 * Kept dependency-free on purpose: these values are needed by BOTH the
 * settings stores and the pure legacy-settings migration chain
 * (`src/migrations/legacySettingsChain.ts`), and the migration chain's module
 * graph must stay free of stores, UI components, the API client and the
 * navigation service. `lib/constants.ts` re-exports them for existing
 * importers.
 */

export const defaultQueryLabel = '*'

// One-time system-suggested user prompts, injected once into userPromptHistory
// (for both fresh installs and upgrades). See settings store version 20 migration.
export const suggestedUserPrompts: string[] = [
  'Ignore the `References Section Format` instruction in the system prompt, and do not include a `References` section in the response.',
  'For inline citations, use the footnote marker syntax `[^1]`, where the `^` preceding the identifier indicates a footnote reference. When multiple citations are required at a single location, each ID should be enclosed in separate footnote markers (e.g., `[^1][^2][^3]`).'
]
