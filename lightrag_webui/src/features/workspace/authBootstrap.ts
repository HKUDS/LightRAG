import type { AuthStatusResponse } from '@/api/lightrag'
import { useAuthStore } from '@/stores/state'
import { activateLoginIdentityFromToken } from '@/lib/loginIdentity'

/**
 * Reconcile the workspace session with what `/auth-status` reports, at the
 * one point this entry contacts the server on an ALREADY-authenticated boot.
 *
 * Two cases, in this order:
 *
 * 1. **Authentication is disabled server-side.** Whatever token this browser
 *    still holds is stale by definition — a leftover named-user token stays
 *    LOCALLY valid until its own expiry, so the router admits it and the
 *    welcome page never runs. Keeping it would leave that user's name and
 *    workspace history on a session that is really a guest's, and its
 *    Authorization header on the wire until the first 401 silently swaps in
 *    a guest token behind the retry. So activate the fresh guest token from
 *    the status response through the SHARED identity cleanup — a guest
 *    replacing a named user is an identity change like any other.
 *
 *    This is NOT the auto-guest activation the workspace entry withholds:
 *    it only runs for a session already admitted by a locally valid token
 *    (an unauthenticated visitor is routed to /welcome, whose explicit
 *    action remains the only way in).
 *
 * 2. **Otherwise**, refresh the stored token's session metadata (versions,
 *    deployment title/description) in place. The guest flag can only be
 *    turned ON here, never off: a guest token stays a guest token.
 */
export function activateSessionFromAuthStatus(
  status: AuthStatusResponse,
  storedToken: string | null
): void {
  if (!status.auth_configured && status.access_token) {
    activateLoginIdentityFromToken(status.access_token)
    useAuthStore
      .getState()
      .login(
        status.access_token,
        true,
        status.core_version,
        status.api_version,
        status.webui_title || null,
        status.webui_description || null
      )
    return
  }

  if (
    storedToken &&
    (status.core_version ||
      status.api_version ||
      status.webui_title ||
      status.webui_description)
  ) {
    const isGuest = status.auth_mode === 'disabled' || useAuthStore.getState().isGuestMode
    useAuthStore
      .getState()
      .login(
        storedToken,
        isGuest,
        status.core_version,
        status.api_version,
        status.webui_title || null,
        status.webui_description || null
      )
  }
}
