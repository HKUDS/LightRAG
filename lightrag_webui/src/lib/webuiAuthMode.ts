/** WebUI capability discovery, not a backend credential-acceptance policy. */
export type WebUIAuthMode = 'account' | 'guest' | 'api-key-only' | 'unknown'

export function webuiAuthMode(status: {
  auth_configured: boolean
  api_key_configured?: boolean
}): WebUIAuthMode {
  if (status.auth_configured) return 'account'
  if (status.api_key_configured === true) return 'api-key-only'
  if (status.api_key_configured === false) return 'guest'
  // Older servers cannot distinguish fully open from API-key-only. Never
  // activate a guest session based on missing discovery information.
  return 'unknown'
}
