import type { QuerySettings } from '@/stores/querySettings'

/**
 * Settings the workspace entry overrides instead of inheriting.
 *
 * Both are debug outlets that make the server return context or a prompt
 * INSTEAD of an answer, which would leave the workspace chat with nothing to
 * render. Everything else is inherited verbatim — see
 * `workspaceQuerySettingsSnapshot`.
 *
 * Adding a key here narrows an agreed contract; do not do it to make a new
 * setting "safer" by default.
 */
export const WORKSPACE_CLAMPED_SETTINGS = {
  only_need_context: false,
  only_need_prompt: false
} as const satisfies Partial<QuerySettings>

/**
 * The workspace entry's query settings, derived from the ones /webui saved.
 *
 * CONTRACT: the workspace entry uses /webui's configuration. Every setting is
 * inherited verbatim except the keys in `WORKSPACE_CLAMPED_SETTINGS`, so a
 * setting added to `QuerySettings` reaches this entry with no change here.
 *
 * Extracted from the component so the contract has one testable definition:
 * a test that re-implemented this spread would keep passing while the real
 * snapshot drifted away from it.
 *
 * Note the inheritance travels through localStorage, so it is per-browser, not
 * per-account: a workspace user who never opened /webui gets
 * `defaultQuerySettings`.
 */
export function workspaceQuerySettingsSnapshot(
  stored: QuerySettings
): QuerySettings {
  return { ...stored, ...WORKSPACE_CLAMPED_SETTINGS }
}
