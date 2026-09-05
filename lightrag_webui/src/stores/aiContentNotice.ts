import { create } from 'zustand'

/**
 * Whether every answer is labelled as AI-generated in the two query UIs
 * (the admin retrieval panel and the `/workspace` query entry).
 *
 * Deployment-level configuration owned by the server
 * (`ENABLE_AI_CONTENT_NOTICE`), delivered on the same responses that already
 * carry the deployment title — `/auth-status` and `/login` (both entries fetch
 * one of them at boot) and `/health` (the admin shell's poll, which keeps the
 * flag fresh across a server restart within one page load).
 *
 * NEVER persisted to localStorage: a deployment that turns the notice off must
 * not keep showing it from a stale cache, and a deployment that turns it on
 * must not have it suppressed by one. The flag is only ever needed once an
 * answer exists, by which time the boot request has long settled, so nothing
 * is gained by caching it across page loads.
 *
 * Older servers omit the field entirely; `applyAiContentNoticeFlag` then
 * leaves the current value untouched (see below), and the default keeps the
 * chat exactly as it was before this option existed.
 */
interface AiContentNoticeState {
  enabled: boolean
  setEnabled: (enabled: boolean) => void
}

export const useAiContentNoticeStore = create<AiContentNoticeState>()((set) => ({
  enabled: false,
  setEnabled: (enabled: boolean) => set({ enabled })
}))

/**
 * Adopt the flag from a server response.
 *
 * Only a real boolean is applied. Everything else — a server too old to send
 * the field, and the hard-coded fallbacks `getAuthStatus` returns when the
 * request fails or answers with HTML — must leave the last known value alone:
 * a failed poll is not a statement that the notice was turned off.
 */
export const applyAiContentNoticeFlag = (value: unknown): void => {
  if (typeof value !== 'boolean') return
  if (useAiContentNoticeStore.getState().enabled === value) return
  useAiContentNoticeStore.getState().setEnabled(value)
}
