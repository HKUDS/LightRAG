/**
 * One-time split of the legacy `settings-storage` envelope into the
 * `lightrag::…` keys (see `lib/storageKeys.ts`).
 *
 * Field mapping is FIXED and exhaustive — it is a spec, not a convention:
 *
 *   legacy settings-storage.querySettings    → lightrag::query-settings-storage
 *   legacy settings-storage.retrievalHistory → lightrag::webui-retrieval-history
 *   (no source)                              → lightrag::workspace-retrieval-history = []
 *
 * The legacy retrievalHistory records the ADMIN entry's debugging questions;
 * copying it into the workspace history would show (and, in bypass mode, send
 * to the LLM) the admin's debug context — exactly the defect the split
 * removes. Blanking both would break the no-data-loss promise. Hence the
 * one-sided mapping above.
 *
 * Semantics (each numbered rule is pinned by tests):
 *  1. Version normalization first — any envelope <= v21 is normalized through
 *     the REUSED legacy chain (`legacySettingsChain.ts`) before splitting.
 *  2. The chain lives in a pure module so importing it here cannot create or
 *     hydrate the very store being split.
 *  3. version > cap (rollback-then-upgrade, unknown future): nothing is read,
 *     cleaned, or overwritten; new keys start from defaults.
 *  4. Migration values are written ONLY for target keys that do not exist;
 *     an existing new key always wins and is never overwritten by old data.
 *  5. ALL new keys are written first; only then are the migrated fields
 *     removed from the legacy envelope and its version bumped. Cleaning
 *     before copying would lose data on a mid-run crash.
 *  6. Partial failure keeps the uncleaned legacy fields; the next startup
 *     retries under the same rules and converges (rule 4 makes re-runs safe).
 *  7. localStorage has no multi-key transaction, so a partially-written state
 *     can physically exist; the app-side guarantee is that it is never
 *     OBSERVED: when this migration did not complete, dependent stores must
 *     not hydrate (they consult `getSettingsMigrationError()` and skip), and
 *     the entry shows a retryable error instead of running on defaults.
 *  8. Both entries run this migration unconditionally; the /workspace entry
 *     may write ALL target keys here (runtime writes to
 *     query-settings-storage remain forbidden for it).
 *
 * No locks and no cross-tab coordination are needed: the target keys are
 * fixed, every executor writes only missing keys, a present key is
 * authoritative, and cleanup happens only after all copies exist — any
 * interleaving of new-code tabs converges to the same result. Tabs still
 * running OLD JavaScript after the deployment may keep rewriting the legacy
 * envelope; those post-migration edits are explicitly allowed to be lost:
 * later runs only clean such rewrites, never merge them into the new keys.
 */

import {
  LEGACY_SETTINGS_VERSION_CAP,
  migrateLegacySettingsState
} from './legacySettingsChain'
import {
  LEGACY_SETTINGS_STORAGE_KEY,
  QUERY_SETTINGS_STORAGE_KEY,
  WEBUI_RETRIEVAL_HISTORY_KEY,
  WORKSPACE_RETRIEVAL_HISTORY_KEY
} from '@/lib/storageKeys'

/** Envelope version of `settings-storage` after the split. */
export const SETTINGS_STORAGE_VERSION_AFTER_SPLIT = 22

/** Envelope version of the freshly split `lightrag::…` persist keys. */
export const QUERY_SETTINGS_STORE_VERSION = 1
export const RETRIEVAL_HISTORY_STORE_VERSION = 1

let migrationError: unknown = null

/**
 * Non-null when the last `runSettingsStorageSplitMigration` did not complete.
 * Dependent stores consult this to skip hydration (rule 7), and the entry
 * shells render a retryable error instead of the app.
 */
export function getSettingsMigrationError(): unknown {
  return migrationError
}

/** Test hook: clear the recorded error between runs. */
export function resetSettingsMigrationErrorForTests(): void {
  migrationError = null
}

/**
 * True when the legacy envelope was written by a NEWER client (version above
 * this build's 22). Rule 3's "never read, clean or overwrite" extends to the
 * settings store itself: hydrating such an envelope would restamp it as v22
 * (zustand persists the migrate result back), and any later set() would
 * clobber fields this build does not know. The settings store consults this
 * on EVERY storage operation (see stores/settings.ts) — the envelope can
 * also arrive mid-session from a newer client's tab — and diverts writes to
 * a session-only fallback while one is present.
 */
export function hasFutureSettingsEnvelope(
  storage: Storage | null = typeof localStorage === 'undefined' ? null : localStorage
): boolean {
  if (!storage) return false
  try {
    const raw = storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)
    if (raw == null) return false
    const version: unknown = JSON.parse(raw)?.version
    return typeof version === 'number' && version > SETTINGS_STORAGE_VERSION_AFTER_SPLIT
  } catch {
    // Unreadable / corrupt envelope: not evidence of a newer client.
    return false
  }
}

function writeIfAbsent(storage: Storage, key: string, value: unknown): void {
  if (storage.getItem(key) == null) {
    storage.setItem(key, JSON.stringify(value))
  }
}

export function runSettingsStorageSplitMigration(
  storage: Storage | null = typeof localStorage === 'undefined' ? null : localStorage
): void {
  migrationError = null
  if (!storage) return
  try {
    const raw = storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)
    if (raw == null) return

    let envelope: any
    try {
      envelope = JSON.parse(raw)
    } catch {
      // Corrupt envelope: treated as "no legacy data" — stores start from
      // defaults; the broken blob is left in place for manual inspection.
      return
    }
    if (!envelope || typeof envelope !== 'object') return

    const version = typeof envelope.version === 'number' ? envelope.version : 0
    if (version > LEGACY_SETTINGS_VERSION_CAP) {
      // Rule 3. Includes our own post-split version (22): nothing left to
      // migrate. Anything higher is unknown — never read, clean or overwrite.
      return
    }

    const state = envelope.state
    if (!state || typeof state !== 'object') return

    // Rule 1: reuse the existing chain to normalize to v21 before splitting.
    const normalized = migrateLegacySettingsState(state, version)

    // Phase 1 (rule 4/5): write every missing target key. JSON.stringify
    // drops `undefined` fields, so an envelope that somehow lacks a source
    // field simply lets the store fall back to its defaults on hydration.
    writeIfAbsent(storage, QUERY_SETTINGS_STORAGE_KEY, {
      state: { querySettings: normalized.querySettings },
      version: QUERY_SETTINGS_STORE_VERSION
    })
    writeIfAbsent(storage, WEBUI_RETRIEVAL_HISTORY_KEY, {
      state: { history: normalized.retrievalHistory ?? [] },
      version: RETRIEVAL_HISTORY_STORE_VERSION
    })
    writeIfAbsent(storage, WORKSPACE_RETRIEVAL_HISTORY_KEY, {
      // Fixed mapping: the workspace history has NO legacy source.
      state: { history: [] },
      version: RETRIEVAL_HISTORY_STORE_VERSION
    })

    // Phase 2 (rule 5): all targets exist — strip the migrated fields from
    // the legacy envelope and bump its version.
    delete normalized.querySettings
    delete normalized.retrievalHistory
    storage.setItem(
      LEGACY_SETTINGS_STORAGE_KEY,
      JSON.stringify({
        ...envelope,
        state: normalized,
        version: SETTINGS_STORAGE_VERSION_AFTER_SPLIT
      })
    )
  } catch (error) {
    // Rule 6/7: keep whatever legacy state remains; record the failure so
    // dependent stores refuse to hydrate and the entry can offer a retry.
    migrationError = error
  }
}
