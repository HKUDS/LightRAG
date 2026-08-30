/**
 * "The auth status was already fetched in THIS page load."
 *
 * Its only job is to stop the app shell from repeating the `/auth-status`
 * request the login/welcome page just made before routing into it — an SPA
 * navigation, so module state is exactly the right lifetime.
 *
 * It deliberately does NOT live in sessionStorage any more. That storage is
 * scoped to the TAB, so the flag survived reloads and suppressed the shell's
 * whole reconciliation step for the rest of the tab's life: a server that
 * changed from auth-enabled to auth-disabled while an unexpired named-user
 * token sat in the browser would be met with that user's identity and
 * history on every reload, because the fresh guest token that replaces the
 * stale one is only read from a status response the shell never requested.
 * Caching the version metadata must never cache the auth reconciliation.
 */

let checkedThisPageLoad = false

/** Record that a login/welcome flow already fetched the auth status. */
export function markVersionCheckedFromLogin(): void {
  checkedThisPageLoad = true
}

/** Whether the status was already fetched since this document loaded. */
export function wasVersionCheckedThisPageLoad(): boolean {
  return checkedThisPageLoad
}

/** Test seam: a real page load starts a fresh module instance. */
export function resetVersionCheckCache(): void {
  checkedThisPageLoad = false
}
