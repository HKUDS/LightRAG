import { create } from 'zustand'
import { checkHealth, type LightragStatus } from '@/api/lightrag'
import { healthCheckInterval } from '@/lib/constants'
import { createSelectors } from '@/lib/utils'
import { useSettingsStore } from './settings'

interface BackendState {
  health: boolean
  message: string | null
  messageTitle: string | null
  status: LightragStatus | null
  lastCheckTime: number
  pipelineBusy: boolean
  healthCheckIntervalId: ReturnType<typeof setInterval> | null
  healthCheckFunction: (() => void) | null
  healthCheckIntervalValue: number

  check: () => Promise<boolean>
  clear: () => void
  setErrorMessage: (message: string, messageTitle: string) => void
  setPipelineBusy: (busy: boolean) => void
  setHealthCheckFunction: (fn: () => void) => void
  resetHealthCheckTimer: () => void
  resetHealthCheckTimerDelayed: (delayMs: number) => void
  clearHealthCheckTimer: () => void
}

interface AuthState {
  isAuthenticated: boolean
  isGuestMode: boolean // Add guest mode flag
  coreVersion: string | null
  apiVersion: string | null
  username: string | null // login username
  webuiTitle: string | null // Custom title
  webuiDescription: string | null // Title description

  login: (
    token: string,
    isGuest?: boolean,
    coreVersion?: string | null,
    apiVersion?: string | null,
    webuiTitle?: string | null,
    webuiDescription?: string | null
  ) => void
  logout: () => void
  setVersion: (coreVersion: string | null, apiVersion: string | null) => void
  setCustomTitle: (webuiTitle: string | null, webuiDescription: string | null) => void
}

const useBackendStateStoreBase = create<BackendState>()((set, get) => ({
  health: true,
  message: null,
  messageTitle: null,
  lastCheckTime: Date.now(),
  status: null,
  pipelineBusy: false,
  healthCheckIntervalId: null,
  healthCheckFunction: null,
  healthCheckIntervalValue: healthCheckInterval * 1000, // Use constant from lib/constants

  check: async () => {
    const health = await checkHealth()
    if (health.status === 'healthy') {
      // Update version information if health check returns it
      if (health.core_version || health.api_version) {
        useAuthStore.getState().setVersion(health.core_version || null, health.api_version || null)
      }

      // Update custom title information if health check returns it
      if ('webui_title' in health || 'webui_description' in health) {
        useAuthStore
          .getState()
          .setCustomTitle(
            'webui_title' in health ? (health.webui_title ?? null) : null,
            'webui_description' in health ? (health.webui_description ?? null) : null
          )
      }

      // Extract and store backend max graph nodes limit
      if (health.configuration?.max_graph_nodes) {
        const maxNodes = Number.parseInt(health.configuration.max_graph_nodes, 10)
        if (!Number.isNaN(maxNodes) && maxNodes > 0) {
          const currentBackendMaxNodes = useSettingsStore.getState().backendMaxGraphNodes

          // Only update if the backend limit has actually changed
          if (currentBackendMaxNodes !== maxNodes) {
            useSettingsStore.getState().setBackendMaxGraphNodes(maxNodes)

            // Auto-adjust current graphMaxNodes if it exceeds the new backend limit
            const currentMaxNodes = useSettingsStore.getState().graphMaxNodes
            if (currentMaxNodes > maxNodes) {
              useSettingsStore.getState().setGraphMaxNodes(maxNodes, true)
            }
          }
        }
      }

      set({
        health: true,
        message: null,
        messageTitle: null,
        lastCheckTime: Date.now(),
        status: health,
        pipelineBusy: health.pipeline_busy,
      })
      return true
    }
    set({
      health: false,
      message: health.message,
      messageTitle: 'Backend Health Check Error!',
      lastCheckTime: Date.now(),
      status: null,
    })
    return false
  },

  clear: () => {
    set({ health: true, message: null, messageTitle: null })
  },

  setErrorMessage: (message: string, messageTitle: string) => {
    set({ health: false, message, messageTitle })
  },

  setPipelineBusy: (busy: boolean) => {
    set({ pipelineBusy: busy })
  },

  setHealthCheckFunction: (fn: () => void) => {
    set({ healthCheckFunction: fn })
  },

  resetHealthCheckTimer: () => {
    const { healthCheckIntervalId, healthCheckFunction, healthCheckIntervalValue } = get()
    if (healthCheckIntervalId) {
      clearInterval(healthCheckIntervalId)
    }
    if (healthCheckFunction) {
      healthCheckFunction() // run health check immediately
      const newIntervalId = setInterval(healthCheckFunction, healthCheckIntervalValue)
      set({ healthCheckIntervalId: newIntervalId })
    }
  },

  resetHealthCheckTimerDelayed: (delayMs: number) => {
    setTimeout(() => {
      get().resetHealthCheckTimer()
    }, delayMs)
  },

  clearHealthCheckTimer: () => {
    const { healthCheckIntervalId } = get()
    if (healthCheckIntervalId) {
      clearInterval(healthCheckIntervalId)
      set({ healthCheckIntervalId: null })
    }
  },
}))

const useBackendState = createSelectors(useBackendStateStoreBase)

export { useBackendState }

const parseTokenPayload = (token: string): { sub?: string; role?: string } => {
  try {
    // JWT tokens are in the format: header.payload.signature
    const parts = token.split('.')
    if (parts.length !== 3) return {}
    const payload = JSON.parse(atob(parts[1]))
    return payload
  } catch (e) {
    console.error('Error parsing token payload:', e)
    return {}
  }
}

const getUsernameFromToken = (token: string): string | null => {
  const payload = parseTokenPayload(token)
  return payload.sub || null
}

const isGuestToken = (token: string): boolean => {
  const payload = parseTokenPayload(token)
  return payload.role === 'guest'
}

const initAuthState = (): {
  isAuthenticated: boolean
  isGuestMode: boolean
  coreVersion: string | null
  apiVersion: string | null
  username: string | null
  webuiTitle: string | null
  webuiDescription: string | null
} => {
  const token = localStorage.getItem('LIGHTRAG-API-TOKEN')
  const coreVersion = localStorage.getItem('LIGHTRAG-CORE-VERSION')
  const apiVersion = localStorage.getItem('LIGHTRAG-API-VERSION')
  const webuiTitle = localStorage.getItem('LIGHTRAG-WEBUI-TITLE')
  const webuiDescription = localStorage.getItem('LIGHTRAG-WEBUI-DESCRIPTION')
  const username = token ? getUsernameFromToken(token) : null

  if (!token) {
    return {
      isAuthenticated: false,
      isGuestMode: false,
      coreVersion: coreVersion,
      apiVersion: apiVersion,
      username: null,
      webuiTitle: webuiTitle,
      webuiDescription: webuiDescription,
    }
  }

  return {
    isAuthenticated: true,
    isGuestMode: isGuestToken(token),
    coreVersion: coreVersion,
    apiVersion: apiVersion,
    username: username,
    webuiTitle: webuiTitle,
    webuiDescription: webuiDescription,
  }
}

export const useAuthStore = create<AuthState>((set) => {
  // Get initial state from localStorage
  const initialState = initAuthState()

  return {
    isAuthenticated: initialState.isAuthenticated,
    isGuestMode: initialState.isGuestMode,
    coreVersion: initialState.coreVersion,
    apiVersion: initialState.apiVersion,
    username: initialState.username,
    webuiTitle: initialState.webuiTitle,
    webuiDescription: initialState.webuiDescription,

    login: (
      token,
      isGuest = false,
      coreVersion = null,
      apiVersion = null,
      webuiTitle = null,
      webuiDescription = null
    ) => {
      localStorage.setItem('LIGHTRAG-API-TOKEN', token)

      if (coreVersion) {
        localStorage.setItem('LIGHTRAG-CORE-VERSION', coreVersion)
      }
      if (apiVersion) {
        localStorage.setItem('LIGHTRAG-API-VERSION', apiVersion)
      }

      if (webuiTitle) {
        localStorage.setItem('LIGHTRAG-WEBUI-TITLE', webuiTitle)
      } else {
        localStorage.removeItem('LIGHTRAG-WEBUI-TITLE')
      }

      if (webuiDescription) {
        localStorage.setItem('LIGHTRAG-WEBUI-DESCRIPTION', webuiDescription)
      } else {
        localStorage.removeItem('LIGHTRAG-WEBUI-DESCRIPTION')
      }

      const username = getUsernameFromToken(token)
      set({
        isAuthenticated: true,
        isGuestMode: isGuest,
        username: username,
        coreVersion: coreVersion,
        apiVersion: apiVersion,
        webuiTitle: webuiTitle,
        webuiDescription: webuiDescription,
      })
    },

    logout: () => {
      localStorage.removeItem('LIGHTRAG-API-TOKEN')

      const coreVersion = localStorage.getItem('LIGHTRAG-CORE-VERSION')
      const apiVersion = localStorage.getItem('LIGHTRAG-API-VERSION')
      const webuiTitle = localStorage.getItem('LIGHTRAG-WEBUI-TITLE')
      const webuiDescription = localStorage.getItem('LIGHTRAG-WEBUI-DESCRIPTION')

      set({
        isAuthenticated: false,
        isGuestMode: false,
        username: null,
        coreVersion: coreVersion,
        apiVersion: apiVersion,
        webuiTitle: webuiTitle,
        webuiDescription: webuiDescription,
      })
    },

    setVersion: (coreVersion, apiVersion) => {
      // Update localStorage
      if (coreVersion) {
        localStorage.setItem('LIGHTRAG-CORE-VERSION', coreVersion)
      }
      if (apiVersion) {
        localStorage.setItem('LIGHTRAG-API-VERSION', apiVersion)
      }

      // Update state
      set({
        coreVersion: coreVersion,
        apiVersion: apiVersion,
      })
    },

    setCustomTitle: (webuiTitle, webuiDescription) => {
      // Update localStorage
      if (webuiTitle) {
        localStorage.setItem('LIGHTRAG-WEBUI-TITLE', webuiTitle)
      } else {
        localStorage.removeItem('LIGHTRAG-WEBUI-TITLE')
      }

      if (webuiDescription) {
        localStorage.setItem('LIGHTRAG-WEBUI-DESCRIPTION', webuiDescription)
      } else {
        localStorage.removeItem('LIGHTRAG-WEBUI-DESCRIPTION')
      }

      // Update state
      set({
        webuiTitle: webuiTitle,
        webuiDescription: webuiDescription,
      })
    },
  }
})
