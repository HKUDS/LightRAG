import { create } from 'zustand'
import { useWebuiRetrievalHistoryStore } from '@/stores/webuiRetrievalHistory'
import { useWorkspaceRetrievalHistoryStore } from '@/stores/workspaceRetrievalHistory'
import {
  WEBUI_RETRIEVAL_HISTORY_KEY,
  WORKSPACE_RETRIEVAL_HISTORY_KEY
} from '@/lib/storageKeys'

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
 */

const PREVIOUS_USER_KEY = 'LIGHTRAG-PREVIOUS-USER'

/** The identity a token asserts: its JWT `sub`, or "guest" when unreadable
 * (only guest activations call this without knowing the name up front). */
export function loginIdentityFromToken(token: string): string {
  try {
    const parts = token.split('.')
    if (parts.length === 3) {
      const sub: unknown = JSON.parse(atob(parts[1]))?.sub
      if (typeof sub === 'string' && sub) return sub
    }
  } catch {
    // Unparseable token: fall through to the guest fallback below.
  }
  return 'guest'
}

/**
 * Record `identity` as the browser's current user, clearing BOTH entries'
 * retrieval histories when it differs from the stored one (a missing stored
 * identity counts as different — e.g. anonymous guest use before the first
 * named login). Returns whether the histories were cleared.
 */
export function applyLoginIdentity(identity: string): boolean {
  let previous: string | null = null
  try {
    previous = localStorage.getItem(PREVIOUS_USER_KEY)
  } catch (error) {
    console.error('Failed to read previous user identity:', error)
  }

  const identityChanged = previous !== identity
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
 * fire only in OTHER tabs and only when the value actually changed, so a
 * matching event is always a genuine identity transition performed
 * elsewhere: clear this tab's LIVE history state (the other tab already
 * cleared the persisted envelopes; clearing again is idempotent) and bump
 * the epoch so mounted query sessions remount empty.
 */
export function handleIdentityStorageEvent(event: {
  key: string | null
  oldValue?: string | null
  newValue?: string | null
}): void {
  if (event.key !== PREVIOUS_USER_KEY) return
  if (event.newValue == null || event.newValue === event.oldValue) return
  useWebuiRetrievalHistoryStore.getState().clearHistory()
  useWorkspaceRetrievalHistoryStore.getState().clearHistory()
  useIdentityEpochStore.setState((state) => ({ epoch: state.epoch + 1 }))
}

/** Installed by BOTH entry bootstraps alongside their other listeners. */
export function watchCrossTabIdentityChanges(): void {
  if (typeof window === 'undefined') return
  window.addEventListener('storage', handleIdentityStorageEvent)
}
