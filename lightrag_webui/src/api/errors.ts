/**
 * Typed marker for a request that ended because the session's authentication
 * is gone (a 401 with no recoverable refresh). By the time this is thrown the
 * API layer has ALREADY navigated to the entry's unauthenticated route, so it
 * signals a session termination, not an answer failure. The query session
 * layer treats the two as separate flows (workspace-entry PRD error split):
 * on this error it only ends the loading state and cleans up request state —
 * it never renders an assistant error bubble and never persists anything
 * error-shaped into retrieval history.
 */
export class AuthenticationRequiredError extends Error {
  constructor(message = 'Authentication required', options?: ErrorOptions) {
    super(message, options)
    this.name = 'AuthenticationRequiredError'
  }
}

/** `instanceof` plus a name check, so the signal survives duplicated module
 * instances (dev HMR, test module registries). */
export function isAuthenticationRequiredError(
  error: unknown
): error is AuthenticationRequiredError {
  return (
    error instanceof AuthenticationRequiredError ||
    (error instanceof Error && error.name === 'AuthenticationRequiredError')
  )
}
