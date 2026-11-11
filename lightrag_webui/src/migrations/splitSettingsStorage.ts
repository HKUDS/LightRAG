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
 *     cleaned, or overwritten; new keys start from defaults. The ONE field
 *     still consulted at a version past the cap is the quota loss marker
 *     (rule 9), and only up to the version this build itself writes (22),
 *     because "nothing left to migrate" and "part of it is gone" are
 *     different answers and the second outranks the first. A STRICTLY future
 *     envelope (> 22) is exempt even from that: it belongs to the newer
 *     client that wrote it, and `hasFutureSettingsEnvelope` handles it.
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
 *  9. An envelope the quota forced this code to shed a field from carries
 *     the loss marker whenever that field existed NOWHERE ELSE — which is
 *     decided per field against the storage as it stands, not by the
 *     field's shape: a scalar always qualifies, and so does an object the
 *     split copies nowhere (`userPromptHistory`) or one whose copy has not
 *     landed yet. No run may proceed over a marked envelope, regardless of
 *     whether there is anything left to migrate, since a shed rung inherits
 *     the version of the payload it replaced and phase 2's is already v22.
 *     Only `acceptSettingsDataLoss`, which only a person calls, clears it.
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

/**
 * Envelope-level marker (outside `state`, so the legacy chain never sees it)
 * saying: this envelope is NOT what the browser used to hold — the write-back
 * ladder below had to shed fields, or everything, to make it fit.
 *
 * Without it the reduced rungs are a lie. `{state:{}}` at the chain's cap
 * version is a perfectly valid empty envelope, so the moment capacity returns
 * the next run migrates it cleanly — no error, no notice — and the user's
 * theme, language and API key are gone silently, the retry on
 * MigrationErrorScreen having converted a visible failure into an invisible
 * one. A partially shed envelope is the same lie in miniature: it migrates
 * "successfully" while the fields that did not fit are simply absent.
 *
 * A run that reads the marker REFUSES and reports, rather than completing
 * over settings it never had. Only `acceptSettingsDataLoss` clears it, and
 * only because a person chose it — losing an API key is not something to
 * discover later through a rejected query.
 */
const SETTINGS_LOST_MARKER = 'lightragSettingsLostToQuota'

/**
 * Some of what the legacy envelope held is GONE — as opposed to a migration
 * that merely failed and may well succeed on the next load. The two need
 * different words and a different offer on the error screen, so they are
 * different types rather than one flag.
 */
export class SettingsDataLostError extends Error {
  constructor() {
    super(
      'The legacy settings envelope could not be preserved while the browser ' +
        'storage was over quota; some of its contents are unrecoverable.'
    )
    this.name = 'SettingsDataLostError'
  }
}

/** `instanceof` plus a name check, so the signal survives duplicated module
 * instances (dev HMR, test module registries). */
export function isSettingsDataLostError(error: unknown): boolean {
  return (
    error instanceof SettingsDataLostError ||
    (error instanceof Error && error.name === 'SettingsDataLostError')
  )
}

/**
 * The user's explicit acceptance of the loss. Strips the marker so the next
 * run may proceed, and adapts to what actually survived:
 *
 * - fields remain → keep them, and the reload migrates what is left;
 * - nothing remains → drop the key, and the reload starts from defaults.
 *
 * Deliberately an ACTION, never automatic: the whole point of the marker is
 * that no run may quietly complete over settings it never had. Callers reload
 * afterwards, because `skipHydration` was decided when the stores were
 * created — clearing the error alone leaves them un-hydrated for the session.
 */
export function acceptSettingsDataLoss(
  storage: Storage | null = typeof localStorage === 'undefined' ? null : localStorage
): void {
  migrationError = null
  if (!storage) return
  const raw = storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)
  if (raw == null) return
  try {
    const envelope = JSON.parse(raw)
    delete envelope[SETTINGS_LOST_MARKER]
    const state = envelope?.state
    if (state && typeof state === 'object' && Object.keys(state).length > 0) {
      storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, JSON.stringify(envelope))
      return
    }
    storage.removeItem(LEGACY_SETTINGS_STORAGE_KEY)
  } catch {
    // Unparseable, or the un-marked rewrite does not fit either. Either way
    // the key must not keep blocking startup: drop it and begin from
    // defaults, which is what the person just asked for.
    storage.removeItem(LEGACY_SETTINGS_STORAGE_KEY)
  }
}

/**
 * The smallest envelope that still says "there is legacy data here": no
 * state, stamped at the version the chain normalizes to, and marked as a
 * loss. Written only as the last step of the write-back ladder below, when
 * even the reduced bytes will not fit. It forfeits the fields the split does
 * not migrate (theme, language, apiKey) — they are already unwritable at
 * that point — but it keeps the NEXT run from reading "no legacy data" (or,
 * with the marker, "an ordinary empty envelope") and declaring the split
 * done over data it never migrated.
 */
const MINIMAL_LEGACY_ENVELOPE = JSON.stringify({
  state: {},
  version: LEGACY_SETTINGS_VERSION_CAP,
  [SETTINGS_LOST_MARKER]: true
})

/**
 * The rungs between the real bytes and total loss: the same envelope shedding
 * progressively more state, largest field first.
 *
 * Two shapes of saving, in order of what they cost the user:
 *
 *  1. every NON-PRIMITIVE field at once — `querySettings`, `retrievalHistory`
 *     and anything a future version adds are the unbounded ones, while the
 *     settings a user would actually miss (theme, language, apiKey) are
 *     scalars worth a few dozen bytes;
 *  2. then those scalars one at a time, biggest first, keeping at least one.
 *
 * Step 2 exists because step 1 alone leaves a hole. This function's other
 * caller writes a payload the split has ALREADY stripped of both objects, so
 * "drop the non-primitives" finds nothing to drop; without step 2 the ladder
 * fell straight from that payload to the empty envelope and forfeited every
 * scalar, which is exactly the loss step 1 was introduced to prevent.
 *
 * Sizing by SHAPE and by measured length, never by a hard-coded field list,
 * keeps this correct as the legacy state evolves. The version is carried over
 * unchanged so a reduced envelope means what the payload it replaces meant.
 *
 * Whether a rung carries the loss marker is decided PER FIELD, against the
 * storage as it stands, and the rule is the same for objects and scalars:
 * shedding a field is free only when its content still exists SOMEWHERE
 * ELSE. Two ways it does not:
 *
 *  * an object field the split copies nowhere. `userPromptHistory` is one,
 *    and the legacy chain writes it into every normalized envelope — it
 *    lives only here, exactly like theme, language and apiKey do;
 *  * a field the split DOES copy, on a run where the copy has not landed.
 *    `freeLegacyHistory` is reached precisely when the first target write
 *    already failed, so `querySettings` can still be sitting only in this
 *    envelope when the ladder drops it.
 *
 * Treating the object rung as free unconditionally was the defect: the
 * reduced envelope migrated "successfully" on the next reload, the store
 * hydrated defaults for query settings the browser had never handed over,
 * and `acceptSettingsDataLoss` — the whole point of the marker — was never
 * offered. The scalars have nowhere else to go by construction, so their
 * rungs are always marked.
 */

/**
 * The fields the split copies out, and the target key that holds the copy.
 * A field listed here is free to shed once its target EXISTS; one that is
 * not listed has no copy anywhere and is never free.
 *
 * `retrievalHistory` is inert today — both callers strip it before the
 * ladder can see it (phase 2 deletes it, `freeLegacyHistory` blanks it), and
 * its forfeit is governed by the degradation policy and reported through
 * `wasHistoryDroppedDuringMigration` instead. It is listed so the mapping is
 * the same fixed spec as the split's own, not a subset that happens to work.
 */
const MIGRATED_FIELD_TARGETS: Record<string, string> = {
  querySettings: QUERY_SETTINGS_STORAGE_KEY,
  retrievalHistory: WEBUI_RETRIEVAL_HISTORY_KEY
}

function* reducedLegacyPayloads(payload: string, storage: Storage): Generator<string> {
  let parsed: any
  try {
    parsed = JSON.parse(payload)
  } catch {
    return
  }
  const state = parsed?.state
  if (!state || typeof state !== 'object') return

  const scalars: Record<string, unknown> = {}
  let droppedObjects = false
  // Set when at least one dropped object exists ONLY in this envelope, so
  // the rung below has to be reported rather than quietly migrated over.
  let droppedUnsavedObject = false
  for (const [key, value] of Object.entries(state)) {
    if (value !== null && typeof value === 'object') {
      droppedObjects = true
      const target = MIGRATED_FIELD_TARGETS[key]
      if (target === undefined || storage.getItem(target) == null) {
        droppedUnsavedObject = true
      }
      continue
    }
    scalars[key] = value
  }
  const rung = (fields: Record<string, unknown>, lost: boolean) =>
    JSON.stringify(
      lost
        ? { ...parsed, state: fields, [SETTINGS_LOST_MARKER]: true }
        : { ...parsed, state: fields }
    )

  // Objects only. Free when every one of them is already in its own key;
  // a loss the moment one of them is not (see the rule above).
  if (droppedObjects) yield rung(scalars, droppedUnsavedObject)

  // Biggest scalar first: each step buys the most room per field forfeited.
  const byCost = Object.keys(scalars).sort(
    (a, b) => JSON.stringify(scalars[b]).length - JSON.stringify(scalars[a]).length
  )
  const remaining = { ...scalars }
  // Stop at one field: an empty state here would be the minimal envelope
  // without being it, and the ladder's last rung already covers that.
  for (const key of byCost.slice(0, -1)) {
    delete remaining[key]
    yield rung(remaining, true)
  }
}

/**
 * Write the legacy envelope, surviving an engine that charges the value it is
 * about to REPLACE until the write completes. Every caller here SHRINKS the
 * key — phase 2 strips the two migrated fields, `freeLegacyHistory` strips
 * the history — so a quota rejection normally means not "the new value does
 * not fit" but "old and new cannot be charged at once". Removing the key
 * first settles that, and because the payload is never larger than what it
 * replaces, the rewrite can never raise the store's total: the removal
 * cannot be what pushes a store over its quota.
 *
 * It CAN still fail, on a store already over quota by more than the removal
 * frees (roughly the whole legacy history here). Nothing avoids that: at
 * that point the store rejects the payload, and the bytes just removed are
 * larger still. Predicting it is not possible either — a probe write can
 * only certify SLACK, whereas what this needs to know is that the excess is
 * bounded by what the removal releases. Gating on a probe therefore refuses
 * the very cases the removal exists to rescue, and measurably dead-ends far
 * more stores than it saves (a retry that can never succeed pins the user on
 * MigrationErrorScreen).
 *
 * So the removal is attempted, and the write-back below is a ladder from the
 * most faithful restoration to the least, each rung smaller than the last:
 * the bytes actually removed, then the same payload reduced to its scalar
 * settings, then the marked empty envelope. The lower rungs matter because
 * they fit in a window where the others do not — the reduced one still
 * carrying the settings a user would miss — and the last one is what keeps
 * the failure VISIBLE to the next run.
 */
function writeShrunkLegacyEnvelope(storage: Storage, payload: string): void {
  const removed = storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)
  try {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, payload)
    return
  } catch (error) {
    if (!isQuotaExceededError(error)) throw error
  }
  storage.removeItem(LEGACY_SETTINGS_STORAGE_KEY)
  try {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, payload)
  } catch (writeBackError) {
    // Restore the value actually READ FROM THE KEY, not the run's original
    // `raw`: the ladder calls this twice, and after `freeLegacyHistory` the
    // key holds the reduced envelope.
    let restored: string | null = null
    for (const fallback of [
      removed,
      ...reducedLegacyPayloads(payload, storage),
      MINIMAL_LEGACY_ENVELOPE
    ]) {
      if (fallback === null) continue
      try {
        storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, fallback)
        restored = fallback
        break
      } catch {
        // Try the next, smaller rung; the error below is what gets recorded.
      }
    }
    // WHICH rung caught is the difference between "try again later" and
    // "some of these settings are gone". Only the faithful restore leaves the
    // browser holding what it held before, so only it keeps the retryable
    // error; anything lower shed fields, and saying so HERE means the FIRST
    // screen the user sees already tells the truth instead of promising that
    // nothing was lost.
    //
    // `restored === null` is the residual edge: not even the ~60-byte marked
    // envelope fits, so the key is absent and a later run will legitimately
    // read "no legacy data". Nothing can be persisted to say otherwise on a
    // store that full, so this session is the only place the loss can be
    // reported — report it.
    if (restored === null || (restored !== removed && restored.includes(SETTINGS_LOST_MARKER))) {
      throw new SettingsDataLostError()
    }
    throw writeBackError
  }
}

/** Phases 1+2 of the split (rules 4/5); `dropHistory` is the quota
 * degradation: the history copy is replaced with an empty one. */
function writeTargetsAndClean(
  storage: Storage,
  context: { envelope: any; normalized: any },
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
    })
  )
}

/** One degraded attempt (history dropped); returns the error, or null on
 * success. */
function tryWriteDegraded(
  storage: Storage,
  context: { envelope: any; normalized: any },
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
 * envelope — the next run re-normalizes it as a no-op and splits it, and only
 * the already-forfeited history is gone. When even THAT does not fit, the
 * write-back ladder sheds further and more than the history goes; it marks
 * what it sheds, so the next run refuses instead of completing over it.
 */
function freeLegacyHistory(
  storage: Storage,
  context: { envelope: any; normalized: any }
): boolean {
  if (context.normalized.retrievalHistory === undefined) return false
  const payload = JSON.stringify({
    ...context.envelope,
    state: { ...context.normalized, retrievalHistory: undefined },
    version: LEGACY_SETTINGS_VERSION_CAP
  })
  writeShrunkLegacyEnvelope(storage, payload)
  delete context.normalized.retrievalHistory
  return true
}

export function runSettingsStorageSplitMigration(
  storage: Storage | null = typeof localStorage === 'undefined' ? null : localStorage
): void {
  migrationError = null
  historyDroppedDuringMigration = false
  if (!storage) return
  let parsedContext: { envelope: any; normalized: any } | null = null
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

    // BEFORE the version check, deliberately. A shed rung carries the
    // version of the payload it replaced, so a ladder that ran from PHASE 2
    // leaves a marked envelope stamped 22 — already past the cap. Asking
    // "is there anything left to migrate?" first would answer "no, this is
    // post-split" and return clean, and the reload after capacity came back
    // would hydrate the partial settings with no error and no acceptance:
    // exactly the silent loss the marker exists to prevent, reintroduced by
    // the ordering alone. `acceptSettingsDataLoss` is the only way past it,
    // and only a person calls that.
    //
    // Bounded to the versions this build itself writes, because refusing is
    // NOT where rule 3 ends. The refusal routes the user to the acceptance
    // action, and `acceptSettingsDataLoss` WRITES: on a strictly future
    // envelope it would rewrite a newer client's bytes to strip a field this
    // build does not own, and it would do so INSTEAD of the future-envelope
    // handling (`hasFutureSettingsEnvelope`) that such an envelope is
    // supposed to get. The marker on a v23 envelope is the newer client's to
    // report, in its own build. Nothing writes that combination today; the
    // bound is what keeps it from becoming a rule 3 violation if anything
    // ever does.
    if (
      version <= SETTINGS_STORAGE_VERSION_AFTER_SPLIT &&
      envelope[SETTINGS_LOST_MARKER] === true
    ) {
      migrationError = new SettingsDataLostError()
      return
    }
    if (version > LEGACY_SETTINGS_VERSION_CAP) {
      // Rule 3. Includes our own post-split version (22): nothing left to
      // migrate. Anything higher is unknown — never read, clean or overwrite.
      return
    }

    const state = envelope.state
    if (!state || typeof state !== 'object') return

    // Rule 1: reuse the existing chain to normalize to v21 before splitting.
    const normalized = migrateLegacySettingsState(state, version)

    parsedContext = { envelope, normalized }
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
