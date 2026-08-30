import { create } from 'zustand'
import { useWebuiRetrievalHistoryStore } from '@/stores/webuiRetrievalHistory'
import { useWorkspaceRetrievalHistoryStore } from '@/stores/workspaceRetrievalHistory'
import { initAuthState, useAuthStore } from '@/stores/state'
import {
  WEBUI_RETRIEVAL_HISTORY_KEY,
  WORKSPACE_RETRIEVAL_HISTORY_KEY
} from '@/lib/storageKeys'
import { decodeBase64Url, legacyMojibakeForm } from '@/lib/base64url'

/**
 * Identity-change cleanup shared by EVERY path that activates a login: the
 * login form, the admin entry's auth-disabled auto-guest, and the workspace
 * welcome page's explicit guest activation.
 *
 * The contract: logout and 401 handling deliberately PRESERVE the retrieval
 * histories (navigationService stores the outgoing username in
 * LIGHTRAG-PREVIOUS-USER); the histories are cleared at the NEXT activation
 * iff the incoming identity differs. A guest activation is an identity like
 * any other — its name is the guest token's `sub` (the literal "guest"
 * server-side, matching what navigation stores after a guest logout) — so a
 * browser last used by a named user must never show that user's
 * conversations to a guest, in EITHER entry's history.
 *
 * A browser with NO stored identity yet is the one case where "differs" is
 * not decidable from the marker alone; see `applyLoginIdentity` for what it
 * can and cannot imply.
 */

const PREVIOUS_USER_KEY = 'LIGHTRAG-PREVIOUS-USER'
const TOKEN_STORAGE_KEY = 'LIGHTRAG-API-TOKEN'

/** The identity a guest token asserts (its server-side `sub`), and the
 * fallback for a token whose payload cannot be read. */
const GUEST_IDENTITY = 'guest'

/** The identity a token asserts: its JWT `sub`, or "guest" when unreadable
 * (only guest activations call this without knowing the name up front).
 *
 * The payload MUST be read through `decodeBase64Url`: bare `atob` rejects
 * the `-`/`_` alphabet a JWT is entitled to use, and this function's failure
 * mode is silent — it would hand back the literal "guest" for a named user,
 * and `applyLoginIdentity` would then preserve that user's conversations for
 * whoever came next. */
export function loginIdentityFromToken(token: string): string {
  try {
    const parts = token.split('.')
    if (parts.length === 3) {
      const decoded = decodeBase64Url(parts[1])
      const sub: unknown = decoded === null ? undefined : JSON.parse(decoded)?.sub
      if (typeof sub === 'string' && sub) return sub
    }
  } catch {
    // Unparseable token: fall through to the guest fallback below.
  }
  return GUEST_IDENTITY
}

/**
 * Whether arriving at `identity` from the recorded `previous` is a genuine
 * identity change — the single rule BOTH the same-tab activation and the
 * cross-tab `storage` handler decide on, so a browser cannot answer it one
 * way for its own tab and another way for the tab next to it.
 *
 * A NULL `previous` is the ABSENCE of evidence, not evidence of a change,
 * and it survives only until this build's first activation — every
 * activation path records the marker. So it means one of exactly two things:
 *
 *   * a browser that never ran ANY build — both histories are empty and
 *     clearing them is a no-op either way; or
 *   * a browser upgrading from a build that recorded the marker ONLY on a
 *     login-form submit or a logout. There, a NAMED user could not
 *     accumulate history without also recording it, so a history found
 *     beside a missing marker was written by guest use.
 *
 * An incoming guest is therefore the same identity that owns that history
 * and must not destroy it — least of all on the one load where the storage
 * split has just migrated it. An incoming NAMED user is a real transition
 * (guest use before the first login) and still counts.
 *
 * A marker written by a build that read JWT payloads as latin-1 holds the
 * MANGLED spelling of a non-ASCII name (`ÙØ¸Ø¯Ø¬` for `كظدج`). That is the
 * same person, so it is not a transition — otherwise correcting the decoding
 * would wipe both histories once, on the upgrade load, for exactly the users
 * whose names the fix is for. `applyLoginIdentity` then records the corrected
 * spelling, so the marker self-heals on that same activation.
 */
function isIdentityChange(previous: string | null, identity: string): boolean {
  if (previous === null) return identity !== GUEST_IDENTITY
  if (previous === identity) return false
  return previous !== legacyMojibakeForm(identity)
}

/**
 * Record `identity` as the browser's current user, clearing BOTH entries'
 * retrieval histories when it differs from the stored one. Returns whether
 * the histories were cleared.
 */
export function applyLoginIdentity(identity: string): boolean {
  let previous: string | null = null
  try {
    previous = localStorage.getItem(PREVIOUS_USER_KEY)
  } catch (error) {
    console.error('Failed to read previous user identity:', error)
  }

  const identityChanged = isIdentityChange(previous, identity)
  if (identityChanged) {
    // A different person (or a guest after a named user): neither entry may
    // show the previous identity's conversations.
    //
    // Destruction outranks rollback preservation here: remove the PERSISTED
    // envelopes outright FIRST, bypassing the future-envelope guard. The
    // guard would otherwise DIVERT the stores' clear-writes and leave a
    // NEWER client's envelope holding the previous identity's conversations
    // — and because the identity below is still committed, that newer
    // client would later skip its own cleanup ("identity already matches")
    // and hydrate them. Raw key removal is version-agnostic: no client, old
    // or new, has anything left to hydrate.
    try {
      localStorage.removeItem(WEBUI_RETRIEVAL_HISTORY_KEY)
      localStorage.removeItem(WORKSPACE_RETRIEVAL_HISTORY_KEY)
    } catch (error) {
      console.error('Failed to remove persisted retrieval histories:', error)
    }
    useWebuiRetrievalHistoryStore.getState().clearHistory()
    useWorkspaceRetrievalHistoryStore.getState().clearHistory()
  }

  try {
    localStorage.setItem(PREVIOUS_USER_KEY, identity)
  } catch (error) {
    console.error('Failed to store user identity:', error)
  }
  return identityChanged
}

/**
 * Activate the identity a TOKEN asserts — the entry point every login path
 * uses, because the token is the only thing that says who was actually
 * authenticated.
 *
 * The login form must not pass its typed username here: when authentication
 * is disabled server-side, `/login` returns a guest token whatever was
 * submitted (see `create_token(username="guest", ...)` in the API server).
 * A form that had rendered before the operator disabled auth would then
 * record the TYPED name, and a user typing the previous named identity would
 * look like "same user" — preserving that user's conversations for a session
 * that is really a guest's, and, in bypass mode, resending them to the LLM.
 * Deriving from the response token makes the decision follow the identity
 * that was granted rather than the one that was requested.
 */
export function activateLoginIdentityFromToken(token: string): boolean {
  return applyLoginIdentity(loginIdentityFromToken(token))
}

/**
 * Cross-tab identity epoch. Bumped when ANOTHER tab records a different
 * identity (see below); the query pages key their session view on it, so a
 * remount drops the live component state — `useQuerySession` copies history
 * into component state only at initialization, so clearing the persisted
 * envelope alone would leave an already-open page displaying (and, in bypass
 * mode, resending; on its next write, re-persisting) the previous identity's
 * conversation under the NEW tab's token.
 */
export const useIdentityEpochStore = create<{ epoch: number }>(() => ({ epoch: 0 }))

/**
 * Storage-event handler for cross-tab identity changes. `storage` events
 * fire only in OTHER tabs and only when the value actually changed — but a
 * changed value is not yet a changed IDENTITY: the marker's FIRST write
 * (oldValue null) is the other tab merely recording what this browser
 * already was, and for a guest that is `isIdentityChange`'s absence-of-
 * evidence case. Deciding it here by any other rule would undo, from the
 * neighbouring tab, exactly what `applyLoginIdentity` refuses to do in its
 * own — the workspace tab sitting on the welcome page would clear (and
 * persist empty) the history the split migration had just moved, the moment
 * an admin tab activated its guest.
 *
 * On a real transition: clear this tab's LIVE history state (the other tab
 * already cleared the persisted envelopes; clearing again is idempotent) and
 * bump the epoch so mounted query sessions remount empty.
 */
export function handleIdentityStorageEvent(event: {
  key: string | null
  oldValue?: string | null
  newValue?: string | null
}): void {
  if (event.key === TOKEN_STORAGE_KEY) {
    // The token is the LAST thing a login writes (applyLoginIdentity runs
    // FIRST), so this event — not the identity marker's — is the reliable
    // signal that another tab's auth transition is complete. Rebuild this
    // tab's auth store from storage so its username, guest/authenticated
    // flags and expiry describe the token its own requests will now send
    // (the API layer reads it per-request from localStorage). Covers login,
    // logout (newValue null → unauthenticated, and the workspace router
    // falls back to /welcome) and guest silent refreshes alike.
    useAuthStore.setState(initAuthState())
    return
  }
  if (event.key !== PREVIOUS_USER_KEY) return
  // A removal (newValue null) is not a transition either: logout deliberately
  // PRESERVES the histories, and the next activation decides.
  if (event.newValue == null) return
  if (!isIdentityChange(event.oldValue ?? null, event.newValue)) return
  useWebuiRetrievalHistoryStore.getState().clearHistory()
  useWorkspaceRetrievalHistoryStore.getState().clearHistory()
  useIdentityEpochStore.setState((state) => ({ epoch: state.epoch + 1 }))
}

/** Installed by BOTH entry bootstraps alongside their other listeners. */
export function watchCrossTabIdentityChanges(): void {
  if (typeof window === 'undefined') return
  window.addEventListener('storage', handleIdentityStorageEvent)
}
