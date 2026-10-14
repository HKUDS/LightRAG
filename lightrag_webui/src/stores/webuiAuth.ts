import { create } from 'zustand'
import { webuiAuthMode, type WebUIAuthMode } from '@/lib/webuiAuthMode'

export const useWebUIAuthStore = create<{ mode: WebUIAuthMode }>(() => ({ mode: 'unknown' }))

export function applyWebUIAuthStatus(status: Parameters<typeof webuiAuthMode>[0]): void {
  useWebUIAuthStore.setState({ mode: webuiAuthMode(status) })
}
