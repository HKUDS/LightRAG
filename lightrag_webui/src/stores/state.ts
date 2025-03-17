import { create } from 'zustand'
import { createSelectors } from '@/lib/utils'
import { checkHealth, LightragStatus } from '@/api/lightrag'

interface BackendState {
  health: boolean
  message: string | null
  messageTitle: string | null

  status: LightragStatus | null

  lastCheckTime: number

  check: () => Promise<boolean>
  clear: () => void
  setErrorMessage: (message: string, messageTitle: string) => void
}

interface AuthState {
  isAuthenticated: boolean;
  showLoginModal: boolean;
  login: (token: string) => void;
  logout: () => void;
  setShowLoginModal: (show: boolean) => void;
}

const useBackendStateStoreBase = create<BackendState>()((set) => ({
  health: true,
  message: null,
  messageTitle: null,
  lastCheckTime: Date.now(),
  status: null,

  check: async () => {
    const health = await checkHealth()
    if (health.status === 'healthy') {
      set({
        health: true,
        message: null,
        messageTitle: null,
        lastCheckTime: Date.now(),
        status: health
      })
      return true
    }
    set({
      health: false,
      message: health.message,
      messageTitle: 'Backend Health Check Error!',
      lastCheckTime: Date.now(),
      status: null
    })
    return false
  },

  clear: () => {
    set({ health: true, message: null, messageTitle: null })
  },

  setErrorMessage: (message: string, messageTitle: string) => {
    set({ health: false, message, messageTitle })
  }
}))

const useBackendState = createSelectors(useBackendStateStoreBase)

export { useBackendState }

export const useAuthStore = create<AuthState>(set => ({
  isAuthenticated: !!localStorage.getItem('LIGHTRAG-API-TOKEN'),
  showLoginModal: false,
  login: (token) => {
    localStorage.setItem('LIGHTRAG-API-TOKEN', token);
    set({ isAuthenticated: true, showLoginModal: false });
  },
  logout: () => {
    localStorage.removeItem('LIGHTRAG-API-TOKEN');
    set({ isAuthenticated: false });
  },
  setShowLoginModal: (show) => set({ showLoginModal: show })
}));
