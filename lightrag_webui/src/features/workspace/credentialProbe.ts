import { create } from 'zustand'
import {
  InvalidApiKeyError,
  RequireApiKeError,
  verifyCredentials
} from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { useBackendState } from '@/stores/state'

/**
 * Credential probe for the workspace entry, shared by the shell (startup and
 * after every API-key dialog close) and the query view (a query rejected on
 * credential grounds — the server key can be rotated long after the startup
 * probe succeeded, and this entry exposes no other API-key control).
 *
 * `apiKeyDialogRequests` is a monotonic counter rather than a boolean: the
 * shell opens its dialog on every increment, so two consecutive failures with
 * the IDENTICAL message still reopen it.
 */
export const useCredentialProbeStore = create<{ apiKeyDialogRequests: number }>(
  () => ({ apiKeyDialogRequests: 0 })
)

/** Whether a failure message is about the API key (as opposed to a model,
 * network or server error, which the query view renders normally). */
export function isApiKeyFailure(message: string): boolean {
  return message.includes(InvalidApiKeyError) || message.includes(RequireApiKeError)
}

const requestApiKeyDialog = (): void => {
  useCredentialProbeStore.setState((state) => ({
    apiKeyDialogRequests: state.apiKeyDialogRequests + 1
  }))
}

let probeGeneration = 0

/**
 * Verify the caller's credentials against /auth/verify (NOT /health, which is
 * whitelisted and answers "healthy" unauthenticated). Only the newest probe
 * may act: rapid re-saves can leave an older one in flight whose stale
 * failure must not reopen the dialog after a valid replacement.
 */
export function runCredentialProbe(): void {
  const generation = ++probeGeneration
  void verifyCredentials()
    .then(() => {
      if (generation !== probeGeneration) return
      // Credentials accepted: drop a leftover API-key error so the dialog
      // shows a clean slate the next time it opens.
      const message = useBackendState.getState().message
      if (message && isApiKeyFailure(message)) {
        useBackendState.getState().clear()
      }
    })
    .catch((error) => {
      if (generation !== probeGeneration) return
      const message = errorMessage(error)
      if (isApiKeyFailure(message)) {
        useBackendState.getState().setErrorMessage(message, 'API Key required')
        requestApiKeyDialog()
      }
      // Anything else (network failure, auth termination that already
      // navigated) is not an API-key problem: leave the dialog alone.
    })
}
