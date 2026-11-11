/**
 * localStorage keys split out of the legacy `settings-storage` envelope.
 *
 * Key shape is `lightrag:<ns>:<name>`. `<ns>` is FIXED to the empty string in
 * this phase — hence the double colon, kept on purpose so the key shape stays
 * uniform and unambiguous to parse. A follow-up (client-state partitioning)
 * will change `<ns>` from "always empty" to "derived from apiPrefix"; root
 * and direct-port deployments will then not change a single byte.
 *
 * Write-ownership contract (runtime, i.e. outside the one-time migration):
 * - query-settings-storage: written ONLY by the /webui entry.
 * - webui-retrieval-history: written ONLY by the /webui entry.
 * - workspace-retrieval-history: written ONLY by the /workspace entry.
 * The one-time split migration is the single exception: whichever entry runs
 * it first may write ALL of these keys.
 */

export const STORAGE_NS = ''

export const storageKey = (name: string): string => `lightrag:${STORAGE_NS}:${name}`

export const QUERY_SETTINGS_STORAGE_KEY = storageKey('query-settings-storage')
export const WEBUI_RETRIEVAL_HISTORY_KEY = storageKey('webui-retrieval-history')
export const WORKSPACE_RETRIEVAL_HISTORY_KEY = storageKey('workspace-retrieval-history')

export const LEGACY_SETTINGS_STORAGE_KEY = 'settings-storage'
