import { useWebuiRetrievalHistoryStore } from '@/stores/webuiRetrievalHistory'
import { useWorkspaceRetrievalHistoryStore } from '@/stores/workspaceRetrievalHistory'

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
