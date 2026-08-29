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
import { hasFutureEnvelope } from '@/lib/guardedStorage'
import { isQuotaExceededError } from '@/lib/storageQuota'

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
  return hasFutureEnvelope(
    LEGACY_SETTINGS_STORAGE_KEY,
    SETTINGS_STORAGE_VERSION_AFTER_SPLIT,
    storage
  )
}

/** Writes `value` only when `key` is absent (rule 4), recording the keys
 * THIS run created — the quota retry may reclaim only those. */
function writeIfAbsent(
  storage: Storage,
  key: string,
  value: unknown,
  writtenKeys: Set<string>
): void {
  if (storage.getItem(key) == null) {
    storage.setItem(key, JSON.stringify(value))
    writtenKeys.add(key)
  }
}

let historyDroppedDuringMigration = false

/**
 * True when the last migration run had to DISCARD the legacy retrieval
 * history to fit within the browser's storage quota (see the degradation
 * path below). The entry shells surface a one-time toast from this — the
 * data is a non-critical test-chat record by product decision, but the user
 * still deserves to hear that it was dropped.
 */
export function wasHistoryDroppedDuringMigration(): boolean {
  return historyDroppedDuringMigration
}

/** Scratch key for the headroom probe below: written and removed on the
 * quota-failure path only, and deliberately tiny — its own size must not be
 * what decides the answer. Nothing ever reads it. */
const QUOTA_PROBE_KEY = 'lightrag::q'

/**
 * Whether the store can still accept ANY write.
 *
 * This is the precondition for REMOVING the legacy envelope. Removing a key
 * and writing back something strictly smaller lowers the store's total, so
 * on a store that is merely unable to replace a key in place, the retry must
 * succeed. There is exactly one state where it cannot: a store that is
 * ALREADY over quota — a quota that shrank under existing data (other
 * origins, an eviction-policy change, another tab's writes). Removal is
 * unrecoverable there, because restoring even the exact bytes just removed
 * fails too.
 *
 * The smallest write there is separates the two: it is rejected only in that
 * already-over-quota state. A store with too little headroom for the probe
 * but enough for the payload is judged unsafe, which costs an in-place
 * rewrite this run and nothing else — the caller records the failure and the
 * next reload retries against whatever space has been freed by then.
 */
function hasWriteHeadroom(storage: Storage): boolean {
  try {
    storage.setItem(QUOTA_PROBE_KEY, '')
  } catch {
    return false
  }
  try {
    storage.removeItem(QUOTA_PROBE_KEY)
  } catch {
    // An 11-byte scratch key left behind is harmless; nothing reads it.
  }
  return true
}

/**
 * Write the legacy envelope, surviving an engine that charges the value it is
 * about to REPLACE until the write completes. Every caller here SHRINKS the
 * key — phase 2 strips the two migrated fields, `freeLegacyHistory` strips
 * the history — so a quota rejection normally means not "the new value does
 * not fit" but "old and new cannot be charged at once". Removing the key
 * first settles that.
 *
 * It settles it ONLY while the store has headroom, which is why the removal
 * is gated on `hasWriteHeadroom`. Without that gate an already-over-quota
 * store took the same path and lost the envelope outright: the retry failed,
 * and so did restoring the bytes just removed. The next run then read "no
 * legacy data", returned early with NO recorded error, and the app started
 * on defaults — theme, language, apiKey and querySettings all gone, none of
 * them ever migrated.
 *
 * The remaining cost is a window in which the envelope is absent, and each
 * caller accepts it against a worse alternative: for phase 2 the loss of the
 * very history copy the split just placed (and by then every target key
 * exists — rule 5 — so only the fields the split does NOT migrate are
 * exposed to a crash here); for `freeLegacyHistory` a migration that no
 * reload can ever complete. `raw` goes back if the retry fails anyway — the
 * quota can shrink between the probe and the write — so no path leaves the
 * browser with NO legacy envelope.
 */
function writeShrunkLegacyEnvelope(storage: Storage, payload: string, raw: string): void {
  try {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, payload)
    return
  } catch (error) {
    if (!isQuotaExceededError(error)) throw error
    // Nothing can be written at all: keep the envelope rather than trade it
    // for a retry that cannot succeed and a restore that cannot either.
    if (!hasWriteHeadroom(storage)) throw error
  }
  storage.removeItem(LEGACY_SETTINGS_STORAGE_KEY)
  try {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, payload)
  } catch (writeBackError) {
    try {
      storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, raw)
    } catch {
      // Nothing further is possible; the error below is what gets recorded.
    }
    throw writeBackError
  }
}

/** Phases 1+2 of the split (rules 4/5); `dropHistory` is the quota
 * degradation: the history copy is replaced with an empty one. */
function writeTargetsAndClean(
  storage: Storage,
  context: { raw: string; envelope: any; normalized: any },
  dropHistory: boolean,
  writtenKeys: Set<string>
): void {
  const { envelope, normalized } = context
  // Phase 1 (rule 4/5): write every missing target key. JSON.stringify
  // drops `undefined` fields, so an envelope that somehow lacks a source
  // field simply lets the store fall back to its defaults on hydration.
  //
  // ORDER MATTERS: the small, fixed-size targets go first and the
  // quota-sensitive history copy LAST. A quota failure then leaves the
  // cheap targets already written, so the degraded retry only has to place
  // an empty history — it never has to squeeze in another required write
  // while the oversized copy still occupies the quota.
  writeIfAbsent(
    storage,
    QUERY_SETTINGS_STORAGE_KEY,
    {
      state: { querySettings: normalized.querySettings },
      version: QUERY_SETTINGS_STORE_VERSION
    },
    writtenKeys
  )
  writeIfAbsent(
    storage,
    WORKSPACE_RETRIEVAL_HISTORY_KEY,
    {
      // Fixed mapping: the workspace history has NO legacy source.
      state: { history: [] },
      version: RETRIEVAL_HISTORY_STORE_VERSION
    },
    writtenKeys
  )
  writeIfAbsent(
    storage,
    WEBUI_RETRIEVAL_HISTORY_KEY,
    {
      state: { history: dropHistory ? [] : (normalized.retrievalHistory ?? []) },
      version: RETRIEVAL_HISTORY_STORE_VERSION
    },
    writtenKeys
  )

  // Phase 2 (rule 5): all targets exist — strip the migrated fields from
  // the legacy envelope and bump its version.
  delete normalized.querySettings
  delete normalized.retrievalHistory
  // Legacy envelopes predate the `languageUserSelected` marker, but their
  // `language` field only ever changed through the selector: a persisted
  // NON-DEFAULT value therefore proves an explicit choice and must keep
  // outranking the browser language after the upgrade. A persisted 'en'
  // stays ambiguous (initial default or chosen) and resolves via the
  // browser, like a fresh visit.
  if (!('languageUserSelected' in normalized)) {
    normalized.languageUserSelected =
      typeof normalized.language === 'string' && normalized.language !== 'en'
  }
  // Shrink-safe: this payload has just shed querySettings and the history,
  // so on an engine that cannot replace a key in place the rejection is
  // transient — and paying for it by discarding the history copy phase 1
  // already placed would throw away data that provably fit.
  writeShrunkLegacyEnvelope(
    storage,
    JSON.stringify({
      ...envelope,
      state: normalized,
      version: SETTINGS_STORAGE_VERSION_AFTER_SPLIT
    }),
    context.raw
  )
}

/** One degraded attempt (history dropped); returns the error, or null on
 * success. */
function tryWriteDegraded(
  storage: Storage,
  context: { raw: string; envelope: any; normalized: any },
  writtenKeys: Set<string>
): unknown {
  try {
    writeTargetsAndClean(storage, context, true, writtenKeys)
    return null
  } catch (error) {
    return error
  }
}

/**
 * Last-resort quota reclamation: drop the retrieval history from the LEGACY
 * envelope itself. Returns false when there is nothing there to free.
 *
 * Reached only on the quota path, after the degraded retry failed too — so
 * the history is already being discarded and this frees the bytes that
 * decision releases. It exists because the legacy envelope can be big enough
 * to leave no room for even the FIRST small target write: nothing of ours
 * would have landed, the reclamation above would have nothing to remove, and
 * every retry (this run's and every reload's) would fail identically, pinning
 * the user on MigrationErrorScreen forever.
 *
 * The reduced envelope KEEPS querySettings and is stamped at the normalized
 * cap version, so a crash right here leaves a valid, still-migratable
 * envelope — the next run re-normalizes it as a no-op and splits it. Only the
 * already-forfeited history is gone.
 */
function freeLegacyHistory(
  storage: Storage,
  context: { raw: string; envelope: any; normalized: any }
): boolean {
  if (context.normalized.retrievalHistory === undefined) return false
  const payload = JSON.stringify({
    ...context.envelope,
    state: { ...context.normalized, retrievalHistory: undefined },
    version: LEGACY_SETTINGS_VERSION_CAP
  })
  writeShrunkLegacyEnvelope(storage, payload, context.raw)
  delete context.normalized.retrievalHistory
  return true
}

export function runSettingsStorageSplitMigration(
  storage: Storage | null = typeof localStorage === 'undefined' ? null : localStorage
): void {
  migrationError = null
  historyDroppedDuringMigration = false
  if (!storage) return
  let parsedContext: { raw: string; envelope: any; normalized: any } | null = null
  // Target keys THIS run created; the quota retry may reclaim only those.
  const writtenKeys = new Set<string>()
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

    parsedContext = { raw, envelope, normalized }
    writeTargetsAndClean(storage, parsedContext, false, writtenKeys)
  } catch (error) {
    // Quota degradation (product decision: the retrieval history is a
    // non-critical test-chat record). A legacy history near the quota cannot
    // coexist with its migrated copy, and recording a migration failure
    // would lock the user into MigrationErrorScreen with a retry that can
    // never succeed. Instead, retry the split WITHOUT the history copy —
    // writeIfAbsent keeps whatever targets the first pass already wrote —
    // and record the drop so the entry shells can show a one-time warning.
    if (parsedContext && isQuotaExceededError(error)) {
      try {
        // Reclaim the oversized copy this run just made, so the retry has
        // room for the writes still outstanding (a later write can hit the
        // quota precisely BECAUSE the copy landed). Only a key THIS run
        // created may go: rule 4 keeps a pre-existing history authoritative,
        // and `writeIfAbsent` would otherwise preserve our own oversized
        // copy and make the retry fail identically, forever.
        if (writtenKeys.has(WEBUI_RETRIEVAL_HISTORY_KEY)) {
          storage.removeItem(WEBUI_RETRIEVAL_HISTORY_KEY)
          writtenKeys.delete(WEBUI_RETRIEVAL_HISTORY_KEY)
        }
        const retryError = tryWriteDegraded(storage, parsedContext, writtenKeys)
        if (retryError == null) {
          historyDroppedDuringMigration = true
          return
        }
        if (!isQuotaExceededError(retryError)) {
          migrationError = retryError
          return
        }
        // Still out of room: what this run wrote (possibly nothing — the
        // very first target write can already fail) was not what held the
        // quota. The legacy history is, so free it there and try once more.
        if (!freeLegacyHistory(storage, parsedContext)) {
          migrationError = retryError
          return
        }
        const finalError = tryWriteDegraded(storage, parsedContext, writtenKeys)
        if (finalError != null) {
          migrationError = finalError
          return
        }
        historyDroppedDuringMigration = true
        return
      } catch (retryError) {
        migrationError = retryError
        return
      }
    }
    // Rule 6/7: keep whatever legacy state remains; record the failure so
    // dependent stores refuse to hydrate and the entry can offer a retry.
    migrationError = error
  }
}
